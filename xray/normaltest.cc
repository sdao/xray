#include "normaltest.h"

NormalTest::NormalTest(Xray* xray) : Material(xray->getContext()) {
  freeze();
}

NormalTest* NormalTest::make(Xray* xray, const Node& n) {
  return new NormalTest(xray);
}

std::string NormalTest::getClosestHitPtxFile() const {
  return "PTX_files/normaltest.cu.ptx";
}

std::string NormalTest::getClosestHitProgram() const {
  return "radiance";
}
