#include "mesh.h"
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags

Mesh::Mesh(Xray xray, optix::float3 origin, std::string name) : Geom(xray.getContext()), _origin(origin) {
  readPolyModel(name);

  _geom["vertexBuffer"]->setBuffer(_vertices);
  _geom["normalBuffer"]->setBuffer(_normals);
  _geom["faceIndices"]->setBuffer(_faces);

  freeze();
}

Mesh* Mesh::make(Xray xray, const Node& n) {
  return new Mesh(
    xray,
    n.getFloat3("origin"),
    n.getString("file")
  );
}

unsigned Mesh::getPrimitiveCount() const {
  return _numFaces;
}

std::string Mesh::getPtxFile() const {
  return "PTX_files/mesh.cu.ptx";
}

std::string Mesh::getIsectProgram() const {
  return "meshIntersect";
}

std::string Mesh::getBoundsProgram() const {
  return "meshBounds";
}

void Mesh::readPolyModel(std::string name) {
  // Create an instance of the Importer class
  Assimp::Importer importer;

  // And have it read the given file with some example postprocessing.
  const aiScene* scene = importer.ReadFile(
    name,
    aiProcess_Triangulate
    | aiProcess_JoinIdenticalVertices
    | aiProcess_SortByPType
    | aiProcess_GenNormals
    | aiProcess_PreTransformVertices
    | aiProcess_ValidateDataStructure
  );

  // If the import failed, report it
  if (!scene) {
    throw std::runtime_error(importer.GetErrorString());
  }

  // Now we can access the file's contents.
  if (scene->mNumMeshes > 0) {
    // Process first mesh only right now.
    // TODO: process multiple meshes.
    aiMesh* mesh = scene->mMeshes[0];

    if (!mesh->HasPositions()) {
      throw std::runtime_error("No vertex positions on the mesh");
    }

    if (!mesh->HasNormals()) {
      throw std::runtime_error("No vertex normals on the mesh");
    }

    // Add points.
    _vertices = _ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh->mNumVertices);
    _normals = _ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh->mNumVertices);
    optix::float3* vertexMap = static_cast<optix::float3*>(_vertices->map());
    optix::float3* normalMap = static_cast<optix::float3*>(_normals->map());
    for (size_t i = 0; i < mesh->mNumVertices; ++i) {
      aiVector3D thisPos = mesh->mVertices[i];
      aiVector3D thisNorm = mesh->mNormals[i];

      vertexMap[i] = optix::make_float3(thisPos.x, thisPos.y, thisPos.z) + _origin;
      normalMap[i] = optix::normalize(optix::make_float3(thisNorm.x, thisNorm.y, thisNorm.z));
    }

    // Add faces.
    _numFaces = mesh->mNumFaces;
    _faces = _ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, mesh->mNumFaces);
    optix::int3* faceMap = static_cast<optix::int3*>(_faces->map());
    for (size_t i = 0; i < mesh->mNumFaces; ++i) {
      aiFace face = mesh->mFaces[i];

      // Only add the triangles (we should have a triangulated mesh).
      if (face.mNumIndices == 3) {
        faceMap[i] = optix::make_int3(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
      } else {
        faceMap[i] = optix::make_int3(-1);
      }
    }

    _vertices->unmap();
    _normals->unmap();
    _faces->unmap();
  }
}