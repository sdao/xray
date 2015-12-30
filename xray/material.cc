#include "material.h"
#include "cuda/shared.cuh"

Material::Material(optix::Context ctx)
  : _ctx(ctx), _material(ctx->createMaterial()) {}

Material::~Material() {}

std::string Material::getEvalBSDFLocalProgram() const {
  return "evalBSDFLocal";
}

std::string Material::getEvalPDFLocalProgram() const {
  return "evalPDFLocal";
}

std::string Material::getSampleLocalProgram() const {
  return "sampleLocal";
}

void Material::freeze() {
  int flags = 0;
  if (doesReflect()) {
    flags |= MATERIAL_REFLECT;
  }
  if (shouldDirectIlluminate()) {
    flags |= MATERIAL_DIRECT_ILLUMINATE;
  }
  _material["materialFlags"]->setInt(flags);
  _material->setClosestHitProgram(
    RAY_TYPE_NO_NEXT_EVENT_ESTIMATION,
    _ctx->createProgramFromPTXFile("ptx/hit_nodirect.cu.ptx", "radiance")
  );
  _material->setClosestHitProgram(
    RAY_TYPE_NEXT_EVENT_ESTIMATION,
    _ctx->createProgramFromPTXFile("ptx/hit_direct.cu.ptx", "radiance")
  );
  _material->setClosestHitProgram(
    RAY_TYPE_SHADOW,
    _ctx->createProgramFromPTXFile("ptx/hit_shadow.cu.ptx", "radiance")
  );
  _material["evalBSDFLocal"]->set(
    _ctx->createProgramFromPTXFile(getPtxFile(), getEvalBSDFLocalProgram())
  );
  _material["evalPDFLocal"]->set(
    _ctx->createProgramFromPTXFile(getPtxFile(), getEvalPDFLocalProgram())
  );
  _material["sampleLocal"]->set(
    _ctx->createProgramFromPTXFile(getPtxFile(), getSampleLocalProgram())
  );
  _frozen = true;
}

optix::Material Material::getMaterial() const {
  if (_frozen) {
    return _material;
  }

  return nullptr;
}
