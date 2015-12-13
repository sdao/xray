#include "material.h"
#include "CUDA_files/shared.cuh"

Material::Material(optix::Context ctx) : _ctx(ctx), _material(ctx->createMaterial()) {}

Material::~Material() {}

std::string Material::getAnyHitPtxFile() const {
  return "PTX_files/anyhit.cu.ptx";
}

std::string Material::getAnyHitProgram() const {
  return "anyHit";
}

void Material::freeze() {
  _material->setClosestHitProgram(RAY_TYPE_NORMAL, _ctx->createProgramFromPTXFile(getClosestHitPtxFile(), getClosestHitProgram()));
  _material->setAnyHitProgram(RAY_TYPE_SHADOW, _ctx->createProgramFromPTXFile(getAnyHitPtxFile(), getAnyHitProgram()));
  _frozen = true;
}

optix::Material Material::getMaterial() const {
  if (_frozen) {
    return _material;
  }

  return nullptr;
}
