#pragma once
#define NOMINMAX
#include <cmath>
#include <random>
#include <functional>
#include <optix_world.h>

namespace util {

  /**
   * Fills the specified buffer with random unsigned ints.
   */
  inline void fillRandom(unsigned int* buffer, size_t len) {
    std::random_device rngDev;
    std::mt19937 rng(rngDev());
    std::uniform_int_distribution<unsigned> unsignedDist;
    for (int i = 0; i < len; ++i) {
      buffer[i] = unsignedDist(rng);
    }
  }

  /**
   * Convenience method to access an OptiX buffer's underlying data,
   * automatically mapping and unmapping the buffer.
   */
  template<typename T>
  inline void withMappedBuffer(
    optix::Buffer buffer,
    std::function<void(T*, size_t)> action
  ) {
    RTsize size[3] = {1, 1, 1};
    buffer->getSize(size[0], size[1], size[2]);

    T* mapped = static_cast<T*>(buffer->map());
    action(mapped, size[0] * size[1] * size[2]);
    buffer->unmap();
  }

  template<typename T>
  inline optix::Buffer putUserBuffer(
    optix::Context ctx,
    std::vector<T*>& userPtrs
  ) {
    optix::Buffer buffer = ctx->createBuffer(
      RT_BUFFER_INPUT,
      RT_FORMAT_USER,
      userPtrs.size()
    );
    buffer->setElementSize(sizeof(T));
    util::withMappedBuffer<T>(buffer, [&](T* mapped, size_t len) {
      for (int i = 0; i < len; ++i) {
        mapped[i] = *userPtrs[i];
      }
    });

    return buffer;
  }

}