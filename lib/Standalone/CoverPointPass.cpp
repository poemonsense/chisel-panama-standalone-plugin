#include <set>
#include <filesystem>
#include <fstream>

#include "llvm/Support/FormatVariadic.h"

#include "Standalone/CoverPointPass.h"

namespace mlir::standalone {
using namespace circt::firrtl;

#define GEN_PASS_DEF_COVERPOINTPASS
#include "Standalone/CoverPointPass.h.inc"

Operation *findDefOp(Value v, Operation *consumerOp) {
  auto defOp = v.getDefiningOp();
  if (!defOp || defOp->getNumResults() > 1) {
    OpBuilder builder(consumerOp);
    builder.setInsertionPoint(consumerOp);
    auto defName = builder.getStringAttr("line_cover_dummy");
    defOp = builder.create<WireOp>(consumerOp->getLoc(), v.getType(), defName);
    builder.create<ConnectOp>(defOp->getLoc(), defOp->getResult(0), v);
  }
  return defOp;
};

void annotateCoverPoint(
  Operation *op,
  const std::string &name,
  const std::string &groupName,
  CircuitOp &circuit
) {
  MLIRContext *context = circuit.getContext();
  OpBuilder builder(circuit);

  auto coverAnno = mlir::DictionaryAttr::get(context, {
    builder.getNamedAttr("class", builder.getStringAttr("xfuzz.CoverPointAnnotation")),
    builder.getNamedAttr("name", builder.getStringAttr(name)),
    builder.getNamedAttr("group", builder.getStringAttr(groupName)),
  });
  op->setAttr("annotations", builder.getArrayAttr({coverAnno}));
}

void annotateCoverPoint(
  Operation *op,
  const std::string &groupName,
  CircuitOp &circuit
) {
  std::string name;
  if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
    name = nameAttr.str();
  else
    name = op->getName().getStringRef().str();

  annotateCoverPoint(op, name, groupName, circuit);
}

class CoverPointInfo {
public:
  Operation *op;

  std::string name;
  std::string group;
  std::string modName;

  int width = -1;

  Operation *cover;
};

class ModuleCoverPointInfo {
public:
  std::vector<int> indices;
  Value clock;
  Value reset;
};

class CoverPointPass
  : public impl::CoverPointPassBase<CoverPointPass> {
public:
  void runOnOperation() final;

private:
  // cover points are divided into groups
  std::unordered_map<std::string, std::vector<CoverPointInfo>> coverPoints;

  std::optional<CoverPointInfo> getCoverPointInfo(Operation *op);

  void findFieldInPort(
    std::function<Value(void)> lazyValue,
    StringRef name,
    Type type,
    std::function<bool(Type, StringRef)> fieldCond,
    bool dirCond,
    SmallVectorImpl<Value> &results,
    FModuleOp moduleOp,
    OpBuilder &builder
  );

  void findFieldInPorts(
    SmallVector<PortInfo> &ports,
    std::function<bool(Type, StringRef)> fieldCond,
    Direction direction,
    SmallVectorImpl<Value> &results,
    FModuleOp moduleOp,
    OpBuilder &builder
  );

  Value getClock(FModuleOp moduleOp, OpBuilder &builder);
  Value getReset(FModuleOp moduleOp, OpBuilder &builder);
  std::tuple<bool, Value, Value> getClockAndReset(FModuleOp moduleOp, OpBuilder &builder);

  InstanceOp createExtModule(std::string group, int index, int width, Location loc, CircuitOp circuitOp, OpBuilder &builder);

  // C++/Verilog gen
  inline std::string getExtModuleName(const std::string &groupName, int w) {
    return "CoverPointDPI_w" + std::to_string(w) + "_" + groupName;
  }
  std::string getExtModuleBody(
    const std::string &groupName,
    int w,
    bool isRaw,
    bool isMultibit
  );

  inline std::string getDpicFuncName(const std::string &groupName) {
    return "cover_dpi_" + groupName;
  }

  void generateCoverCppHeader(const std::string &outputDir);
  void generateCoverCpp(const std::string &outputDir);
};

std::unique_ptr<mlir::Pass> createCoverPointPass() {
  return std::make_unique<CoverPointPass>();
}

void registerCoverPointPass() {
  PassRegistration<CoverPointPass>();
}

void CoverPointPass::runOnOperation() {
  CircuitOp circuitOp = getOperation();

  // get all cover points
  circuitOp->walk([&](Operation *op) mutable {
    if (auto cOpt = getCoverPointInfo(op)) {
      coverPoints[cOpt->group].push_back(*cOpt);
    }
  });

  // iterate over each group of cover points
  for (auto &[groupName, points]: coverPoints) {
    std::unordered_map<std::string, ModuleCoverPointInfo> modCoverPoints;

    auto index = 0;
    for (auto c = points.begin(); c != points.end(); ) {
      auto moduleOp = c->op->getParentOfType<FModuleOp>();
      auto builder = OpBuilder::atBlockBegin(moduleOp.getBodyBlock());
      auto isTopLevel = isa<FModuleOp>(c->op->getParentOp());

      // clock, reset
      auto [cached, clock, reset] = getClockAndReset(moduleOp, builder);
      if (!clock || !reset) {
        c = points.erase(c);
        if (!cached) {
          mlir::emitWarning(moduleOp.getLoc()) << "[" << moduleOp.getName()
              << "] clock/reset port not found. Skip all cover points.";
        }
        continue;
      }

      c->modName = moduleOp.getName().str();
      if (modCoverPoints.find(c->modName) == modCoverPoints.end()) {
        modCoverPoints[c->modName] = ModuleCoverPointInfo();
        modCoverPoints[c->modName].clock = clock;
        modCoverPoints[c->modName].reset = reset;
      }
      modCoverPoints[c->modName].indices.push_back(index++);

      Value curValue = c->op->getResult(0);
      auto t = dyn_cast<IntType>(curValue.getType().cast<FIRRTLType>());
      c->width = t.getWidthOrSentinel();
      Location loc = c->op->getLoc();

      // builder must be moved after clock and reset
      auto clockOp = clock.getDefiningOp();
      if (clockOp && builder.getInsertionPoint()->isBeforeInBlock(clockOp)) {
        builder.setInsertionPointAfter(clockOp);
      }
      auto resetOp = reset.getDefiningOp();
      if (resetOp && builder.getInsertionPoint()->isBeforeInBlock(resetOp)) {
        builder.setInsertionPointAfter(resetOp);
      }

      // Create the `x_reg` for cover point `x`
      // val x_reg = RegNext(x)
      auto regName = builder.getStringAttr("cover_reg");
      auto reg = builder.create<RegOp>(loc, t, clock, regName);

      // If the cover point `x` is under some When context,
      // we need to add another xor_reg to capture `x` ^ `x_reg`.
      // val xor_reg = RegNext(x ^ x_reg, 0.U)
      RegResetOp xorReg;
      RegOp xorAsyncReg;
      if (!isTopLevel) {
        auto xorRegName = builder.getStringAttr("xor_reg");
        if (reset.getType().isa<AsyncResetType>()) {
          xorAsyncReg = builder.create<RegOp>(loc, t, clock, xorRegName);
        } else {
          auto zero = builder.create<ConstantOp>(loc, t, APInt(c->width, 0));
          xorReg = builder.create<RegResetOp>(loc, t, clock, reset, zero, xorRegName);
        }
      }

      // create in-place expressions: update x_reg (and xor_reg if necessary)
      builder = OpBuilder::atBlockEnd(c->op->getBlock());

      // x_reg := x
      builder.create<ConnectOp>(loc, reg.getResult(), c->op->getResult(0));

      // val xorVal = cover_reg ^ curValue
      auto xorVal = builder.create<XorPrimOp>(loc, c->op->getResult(0), reg.getResult());
      c->cover = xorVal;

      if (!isTopLevel) {
        // xor_reg := xorVal
        // Chisel feature? For async reset behavior, insert mux: reset ? 0.U : xorVal
        if (reset.getType().isa<AsyncResetType>()) {
          auto zero = builder.create<ConstantOp>(loc, t, APInt(c->width, 0));
          auto mux = builder.create<MuxPrimOp>(loc, reset, zero, xorVal.getResult());
          builder.create<ConnectOp>(loc, xorAsyncReg.getResult(), mux.getResult());
        } else {
          builder.create<ConnectOp>(loc, xorReg.getResult(), xorVal.getResult());
        }
        c->cover = xorReg;
      }

      c++;
    }

    // one DPI-C BlackBox and IO connections for each module
    index = 0;
    for (auto &[modName, modCover] : modCoverPoints) {
      auto moduleOp = circuitOp.lookupSymbol<FModuleOp>(modName);
      auto builder = OpBuilder::atBlockEnd(moduleOp.getBodyBlock());

      // cover values are concatenated
      Value concatVal = nullptr;
      auto totalWidth = 0;
      for (auto idx : modCover.indices) {
        totalWidth += points[idx].width;
        auto cover = points[idx].cover->getResult(0);
        if (!concatVal) {
          concatVal = cover;
        } else {
          concatVal = builder.create<CatPrimOp>(moduleOp.getLoc(), cover, concatVal);
        }
      }

      // connect xorVal (or xor_reg) to the cover point BlackBox
      auto loc = builder.getUnknownLoc();
      auto inst = this->createExtModule(groupName, index, totalWidth, loc, circuitOp, builder);
      builder.create<ConnectOp>(loc, inst.getResult(0), modCover.clock);
      // reset (Reset/AsyncReset) should be converted to UInt
      auto resetUInt = builder.create<AsUIntPrimOp>(loc, modCover.reset);
      builder.create<ConnectOp>(loc, inst.getResult(1), resetUInt.getResult());
      builder.create<ConnectOp>(loc, inst.getResult(2), concatVal);

      index += totalWidth;

      llvm::outs() << "[INFO] Created " << inst.getName() << " in module " << modName
        << " for group: " << groupName << " (" << totalWidth << " bits)\n";
    }

    llvm::outs() << "[INFO] Created " << index << " cover points for group: " << groupName << "\n";
  }

  std::string outputDir = getenv("NOOP_HOME") + std::string("/build/generated-src");
  std::filesystem::create_directories(outputDir);
  generateCoverCppHeader(outputDir);
  generateCoverCpp(outputDir);
}

std::optional<CoverPointInfo> CoverPointPass::getCoverPointInfo(Operation *op) {
  auto attr = op->getAttr("annotations");
  auto arrayAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>();

  if (!arrayAttr)
    return std::nullopt;

  for (auto elem : arrayAttr) {
    auto dict = elem.dyn_cast<mlir::DictionaryAttr>();

    if (!dict)
      continue;

    auto classAttr = dict.getAs<mlir::StringAttr>("class");
    if (!classAttr || !classAttr.getValue().endswith(".CoverPointAnnotation"))
      continue;

    auto nameAttr = dict.getAs<mlir::StringAttr>("name");
    auto groupAttr = dict.getAs<mlir::StringAttr>("group");

    if (!nameAttr || !groupAttr) {
      llvm::errs() << "[ERROR] Annotation found with class = " << classAttr.getValue()
                    << ", but missing 'name' or 'group'\n";
      llvm::errs() << "[ERROR] Full annotation dict: " << dict << "\n";
      llvm::report_fatal_error("getCoverPointInfo: missing 'name' or 'group' field");
    }

    CoverPointInfo info;
    info.op = op;
    info.name = nameAttr.getValue().str();
    // the name may also be implicitly extracted from the operation
    if (info.name == "unknown") {
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
        info.name = nameAttr.getValue().str();
      }
    }
    info.group = groupAttr.getValue().str();

    return info;
  }

  return std::nullopt;
}

void CoverPointPass::findFieldInPort(
  std::function<Value(void)> lazyValue,
  StringRef name,
  Type type,
  std::function<bool(Type, StringRef)> fieldCond,
  bool dirCond,
  SmallVectorImpl<Value> &results,
  FModuleOp moduleOp,
  OpBuilder &builder
) {
  if (fieldCond(type, name)) {
    if (dirCond) {
      results.push_back(lazyValue());
    }
  }
  else if (auto bundle = type.dyn_cast<BundleType>()) {
    for (auto it : llvm::enumerate(bundle.getElements())) {
      size_t index = it.index();
      const auto &elem = it.value();
      bool fieldIsTargetDirection = dirCond ^ elem.isFlip;
      auto subfield = [&]() {
        return builder.create<SubfieldOp>(moduleOp.getLoc(), elem.type, lazyValue(), index);
      };
      findFieldInPort(subfield, elem.name, elem.type, fieldCond, fieldIsTargetDirection, results, moduleOp, builder);
    }
  }
}

void CoverPointPass::findFieldInPorts(
  SmallVector<PortInfo> &ports,
  std::function<bool(Type, StringRef)> fieldCond,
  Direction direction,
  SmallVectorImpl<Value> &results,
  FModuleOp moduleOp,
  OpBuilder &builder
) {
  auto *body = moduleOp.getBodyBlock();
  for (auto it : llvm::enumerate(moduleOp.getPorts())) {
    auto index = it.index();
    auto &port = it.value();
    bool isTargetDir = port.direction == direction;
    auto portVal = [&]() { return body->getArgument(index); };
    findFieldInPort(portVal, port.name, port.type, fieldCond, isTargetDir, results, moduleOp, builder);
  }
}

Value CoverPointPass::getClock(FModuleOp moduleOp, OpBuilder &builder) {
  auto ports = moduleOp.getPorts();

  SmallVector<Value> clockPorts;
  auto clockCond = [](Type t, StringRef name) { return t.isa<ClockType>(); };
  findFieldInPorts(ports, clockCond, Direction::In, clockPorts, moduleOp, builder);

  if (clockPorts.empty())
    return nullptr;

  if (clockPorts.size() > 1) {
    // TODO: raise warning here
  }

  return clockPorts.front();
}

Value CoverPointPass::getReset(FModuleOp moduleOp, OpBuilder &builder) {
  auto ports = moduleOp.getPorts();

  SmallVector<Value> resetPorts;
  auto resetCond = [](Type t, StringRef name) {
    if (t.isa<ResetType>() || t.isa<AsyncResetType>())
      return true;
    if (auto uintType = t.dyn_cast<UIntType>())
      return uintType.getWidthOrSentinel() == 1 && name.endswith("reset");
    return false;
  };
  findFieldInPorts(ports, resetCond, Direction::In, resetPorts, moduleOp, builder);

  if (resetPorts.empty())
    return nullptr;

  if (resetPorts.size() > 1) {
    // TODO: raise warning here
  }

  return resetPorts.front();
}

std::tuple<bool, Value, Value> CoverPointPass::getClockAndReset(FModuleOp moduleOp, OpBuilder &builder) {
  static std::unordered_map<std::string, std::pair<Value, Value>> cache;
  auto cacheKey = moduleOp.getName().str();
  if (auto it = cache.find(cacheKey); it != cache.end()) {
    auto [clock, reset] = it->second;
    return {true, clock, reset};
  }

  auto clock = getClock(moduleOp, builder);
  auto reset = getReset(moduleOp, builder);
  cache[cacheKey] = {clock, reset};
  return {false, clock, reset};
}

InstanceOp CoverPointPass::createExtModule(
  std::string group,
  int index,
  int width,
  Location loc,
  CircuitOp circuitOp,
  OpBuilder &builder
) {
  Block *circuitBlock = &circuitOp.getBody().front();
  OpBuilder::InsertPoint saveIP = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(circuitBlock);

  auto *context = builder.getContext();

  auto extModName = getExtModuleName(group, width);

  auto clock = PortInfo(builder.getStringAttr("clock"), ClockType::get(context), Direction::In);;
  auto reset = PortInfo(builder.getStringAttr("reset"), UIntType::get(context, 1), Direction::In);
  auto valid = PortInfo(builder.getStringAttr("valid"), UIntType::get(context, width), Direction::In);
  SmallVector<circt::firrtl::PortInfo> ports = {clock, reset, valid};

  auto convention = ConventionAttr::get(context, Convention::Internal);

  std::string verilogBody = getExtModuleBody(group, width, false, true);
  auto blackboxInlineAnno = mlir::DictionaryAttr::get(
    context,
    {
      builder.getNamedAttr("class", builder.getStringAttr("firrtl.transforms.BlackBoxInlineAnno")),
      builder.getNamedAttr("name", builder.getStringAttr(extModName + ".sv")),
      builder.getNamedAttr("text", builder.getStringAttr(verilogBody)),
    });
  auto annotations = builder.getArrayAttr({blackboxInlineAnno});

  auto nameAttr = builder.getStringAttr("COVER_INDEX");
  auto intValue = builder.getI32IntegerAttr(index);
  auto intType = intValue.getType();
  auto paramAttr = ParamDeclAttr::get(context, nameAttr, intType, intValue);
  auto parameters = builder.getArrayAttr({paramAttr});

  auto extModule = builder.create<FExtModuleOp>(
    loc,
    builder.getStringAttr(extModName + "_" + std::to_string(index)),
    convention,
    ports,
    extModName,
    annotations,
    parameters,
    ArrayAttr{}
  );
  extModule.setVisibility(SymbolTable::Visibility::Private);

  builder.restoreInsertionPoint(saveIP);

  auto instName = builder.getStringAttr("cover_inst");
  auto inst = builder.create<InstanceOp>(
    loc,
    extModule.getPorts(),
    extModule.getModuleNameAttr(),
    instName
  );

  return inst;
}

std::string CoverPointPass::getExtModuleBody(
  const std::string &groupName,
  int w,
  bool isRaw,
  bool isMultibit
) {
  auto extModName = getExtModuleName(groupName, w);
  auto dpiFuncName = getDpicFuncName(groupName);

  auto w_s = [](int _w) -> std::string {
    return _w > 1 ? "[" + std::to_string(_w - 1) + " : 0] " : "";
  };
  std::string io = llvm::formatv("input clock,\n  input reset,\n  input {0}valid", w_s(w));
  std::string extraCond = (w > 1 || isRaw) ? "" : " && valid";

  std::string funcCall;
  if (isMultibit && w > 1) {
    std::stringstream ss;
    for (int i = 0; i < w; ++i) {
      ss << "      if (valid[" << i << "]) begin\n";
      ss << "        " << dpiFuncName << "(COVER_INDEX + " << i << ");\n";
      ss << "      end\n";
    }
    funcCall = ss.str();
  } else {
    std::string extraIndex = (w > 1 || isRaw) ? " + valid" : "";
    funcCall = "      " + dpiFuncName + "(COVER_INDEX" + extraIndex + ");\n";
  }

  std::stringstream verilog;
  verilog << "/*verilator tracing_off*/\n";
  verilog << "module " << extModName << "(\n  " << io << "\n);\n";
  verilog << "  parameter COVER_INDEX;\n";
  verilog << "`ifndef SYNTHESIS\n";
  verilog << "  import \"DPI-C\" function void " << dpiFuncName << "(longint unsigned cover_index);\n";
  verilog << "  always @(posedge clock) begin\n";
  verilog << "    if (!reset" << extraCond << ") begin\n";
  verilog << funcCall;
  verilog << "    end\n";
  verilog << "  end\n";
  verilog << "`endif\n";
  verilog << "endmodule\n";

  return verilog.str();
}

void CoverPointPass::generateCoverCppHeader(const std::string &outputDir) {
  std::stringstream ss;
  ss << "#ifndef __FIRRTL_COVER_H__\n";
  ss << "#define __FIRRTL_COVER_H__\n\n";
  ss << "#include <cstdint>\n\n";

  ss << "typedef struct {\n";
  ss << "  uint8_t* points;\n";
  ss << "  const uint64_t total;\n";
  ss << "  const char* name;\n";
  ss << "  const char** point_names;\n";
  ss << "} FIRRTLCoverPoint;\n\n";

  ss << "typedef struct {\n";
  ss << "  const FIRRTLCoverPoint cover;\n";
  ss << "  bool is_feedback;\n";
  ss << "} FIRRTLCoverPointParam;\n\n";

  ss << "extern FIRRTLCoverPointParam firrtl_cover[" << coverPoints.size() << "];\n\n";

  ss << "\n#endif // __FIRRTL_COVER_H__\n";

  std::ofstream header(outputDir + "/firrtl-cover.h");
  header << ss.str();
  header.close();
}

void CoverPointPass::generateCoverCpp(const std::string &outputDir) {
  std::stringstream ss;

  ss << "#include \"firrtl-cover.h\"\n\n";

  // Define CoverPoints struct
  ss << "typedef struct {\n";
  for (const auto &[groupName, points] : coverPoints) {
    ss << "  uint8_t " << groupName << "[" << points.size() << "];\n";
  }
  ss << "} CoverPoints;\n";
  ss << "static CoverPoints coverPoints;\n";

  // DPI-C functions
  for (const auto &[groupName, points] : coverPoints) {
    ss << "\nextern \"C\" void " << getDpicFuncName(groupName) << "(uint64_t index) {\n";
    ss << "  coverPoints." << groupName << "[index] = 1;\n";
    ss << "}\n";
  }

  // Names arrays
  for (const auto &[groupName, points] : coverPoints) {
    ss << "\nstatic const char *" << groupName << "_NAMES[] = {\n";
    for (const auto &c : points) {
      for (int i = 0; i < c.width; ++i) {
        std::string suffix = (c.width == 1) ? "" : " == " + std::to_string(i);
        ss << "  \"" << c.modName << "." << c.name << suffix << "\",\n";
      }
    }
    ss << "};\n";
  }

  // firrtl_cover array
  ss << "\nFIRRTLCoverPointParam firrtl_cover[" << coverPoints.size() << "] = {\n";
  int i = 0;
  for (const auto &[groupName, points] : coverPoints) {
    std::string isDefaultFeedback = (i == 0 ? "true" : "false");
    ss << "  { { coverPoints." << groupName << ", " << points.size() << "UL, \"" << groupName << "\", "
       << groupName << "_NAMES }, " << isDefaultFeedback << " },\n";
    ++i;
  }
  ss << "};\n";

  std::ofstream cpp(outputDir + "/firrtl-cover.cpp");
  cpp << ss.str();
  cpp.close();
}

} // namespace mlir::standalone
