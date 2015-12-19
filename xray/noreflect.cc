#include "noreflect.h"

NoReflect::NoReflect(Xray* xray) : Material(xray->getContext()) {
  freeze();
}

NoReflect* NoReflect::make(Xray* xray) {
  return new NoReflect(xray);
}

std::string NoReflect::getPtxFile() const {
  return "ptx/constant.cu.ptx";
}

bool NoReflect::doesReflect() const {
  return false;
}

bool NoReflect::shouldDirectIlluminate() const {
  return false;
}
