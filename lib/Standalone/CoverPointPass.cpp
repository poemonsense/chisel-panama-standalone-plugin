#include <set>
#include <filesystem>
#include <fstream>

#include "llvm/Support/FormatVariadic.h"

#include "Standalone/CoverPointPass.h"

namespace mlir::standalone {
using namespace circt::firrtl;

#define GEN_PASS_DEF_COVERPOINTPASS
#include "Standalone/CoverPointPass.h.inc"

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
  int index = -1;
};

class CoverPointPass
  : public impl::CoverPointPassBase<CoverPointPass> {
public:
  void runOnOperation() final;

private:
  // cover points are divided into groups
  std::unordered_map<std::string, std::vector<CoverPointInfo>> coverPoints;

  std::optional<CoverPointInfo> getCoverPointInfo(Operation *op);
  std::pair<Value, Value> getClockAndReset(Operation *op);
  InstanceOp createExtModule(const CoverPointInfo &c, Location loc, CircuitOp circuitOp, OpBuilder &builder);

  // C++/Verilog gen
  inline std::string getExtModuleName(const CoverPointInfo &c) {
    return getExtModuleName(c.group, c.width);
  }
  inline std::string getExtModuleName(const std::string &groupName, int w) {
    return "CoverPointDPI_w" + std::to_string(w) + "_" + groupName;
  }
  std::string getExtModuleBody(const std::string &groupName, int w);

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
  OpBuilder builder(&getContext());

  // get all cover points
  circuitOp->walk([&](Operation *op) mutable {
    if (auto cOpt = getCoverPointInfo(op)) {
      cOpt->index = coverPoints[cOpt->group].size();
      coverPoints[cOpt->group].push_back(*cOpt);
    }
  });

  // iterate over each group of cover points
  for (auto &[groupName, points]: coverPoints) {
    for (auto &c : points) {
      Value curValue = c.op->getResult(0);
      auto t = dyn_cast<IntType>(curValue.getType().cast<FIRRTLType>());
      c.width = t.getWidthOrSentinel();

      builder.setInsertionPointAfter(c.op);
      Location loc = c.op->getLoc();

      // val cover_reg = RegNext(curValue, 0)
      auto zeroValue = APInt(c.width, 0);
      auto zero = builder.create<ConstantOp>(loc, t, zeroValue);
      auto regName = builder.getStringAttr("cover_reg");
      auto [clock, reset] = getClockAndReset(c.op);
      auto reg = builder.create<RegResetOp>(
          loc, t, clock, reset, zero, regName);
      builder.create<ConnectOp>(loc, reg.getResult(), curValue);

      // val xorVal = cover_reg ^ curValue
      auto xorVal = builder.create<XorPrimOp>(loc, curValue, reg.getResult());

      // connect xorVal to the cover point BlackBox
      auto inst = this->createExtModule(c, loc, circuitOp, builder);
      builder.create<ConnectOp>(loc, inst.getResult(0), clock);
      builder.create<ConnectOp>(loc, inst.getResult(1), reset);
      builder.create<ConnectOp>(loc, inst.getResult(2), xorVal.getResult());
    }
  }

  std::string outputDir = getenv("NOOP_HOME") + std::string("/build/generated-src");
  std::filesystem::create_directories(outputDir);
  generateCoverCppHeader(outputDir);
  generateCoverCpp(outputDir);
}

std::optional<CoverPointInfo> CoverPointPass::getCoverPointInfo(Operation *op) {
  auto attr = op->getAttr("annotations");
  if (auto arrayAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
    for (auto elem : arrayAttr) {
      if (auto dict = elem.dyn_cast<mlir::DictionaryAttr>()) {
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
        info.modName = op->getParentOfType<FModuleOp>().getName().str();
        return info;
      }
    }
  }

  return std::nullopt;
}


std::pair<Value, Value> CoverPointPass::getClockAndReset(Operation *op) {
  auto parentMod = op->getParentOfType<FModuleOp>();
  auto *body = parentMod.getBodyBlock();
  Value clock, reset;

  for (auto [idx, port] : llvm::enumerate(parentMod.getPorts())) {
    if (port.name == "clock")
      clock = body->getArgument(idx);
    else if (port.name == "reset")
      reset = body->getArgument(idx);
  }

  if (!clock || !reset)
    llvm::report_fatal_error("getClockAndReset: clock/reset port not found");

  return {clock, reset};
}

InstanceOp CoverPointPass::createExtModule(const CoverPointInfo &c, Location loc, CircuitOp circuitOp, OpBuilder &builder) {
  Block *circuitBlock = &circuitOp.getBody().front();
  OpBuilder::InsertPoint saveIP = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(circuitBlock);

  auto *context = builder.getContext();

  auto extModName = getExtModuleName(c);

  auto clock = PortInfo(builder.getStringAttr("clock"), ClockType::get(context), Direction::In);;
  auto reset = PortInfo(builder.getStringAttr("reset"), UIntType::get(context, 1), Direction::In);
  auto valid = PortInfo(builder.getStringAttr("valid"), UIntType::get(context, c.width), Direction::In);
  SmallVector<circt::firrtl::PortInfo> ports = {clock, reset, valid};

  auto convention = ConventionAttr::get(context, Convention::Internal);

  std::string verilogBody = getExtModuleBody(c.group, c.width);
  auto blackboxInlineAnno = mlir::DictionaryAttr::get(
    context,
    {
      builder.getNamedAttr("class", builder.getStringAttr("firrtl.transforms.BlackBoxInlineAnno")),
      builder.getNamedAttr("name", builder.getStringAttr(extModName + ".sv")),
      builder.getNamedAttr("text", builder.getStringAttr(verilogBody)),
    });
  auto annotations = builder.getArrayAttr({blackboxInlineAnno});

  auto nameAttr = builder.getStringAttr("COVER_INDEX");
  auto intValue = builder.getI32IntegerAttr(c.index);
  auto intType = intValue.getType();
  auto paramAttr = ParamDeclAttr::get(context, nameAttr, intType, intValue);
  auto parameters = builder.getArrayAttr({paramAttr});

  auto extModule = builder.create<FExtModuleOp>(
    loc,
    builder.getStringAttr(extModName + "_" + std::to_string(c.index)),
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

std::string CoverPointPass::getExtModuleBody(const std::string &groupName, int w) {
  auto extModName = getExtModuleName(groupName, w);
  auto dpiFuncName = getDpicFuncName(groupName);

  auto w_s = [](int _w) -> std::string {
    return _w > 1 ? "[" + std::to_string(_w - 1) + " : 0] " : "";
  };
  std::string io = llvm::formatv("input clock,\n  input reset,\n  input {0}valid", w_s(w));

  auto isRaw = false;
  std::string extraCond = (w > 1 || isRaw) ? "" : " && valid";

  auto isMultibit = false;
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
  verilog << "  import \"DPI-C\" function void " << dpiFuncName << "(longint cover_index);\n";
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
