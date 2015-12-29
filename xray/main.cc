#include <iostream>
#include <SDL.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <chrono>
#include "scene.h"
#include "camera.h"
#include "tinyexr/tinyexr.h"

using boost::format;

void writeToEXR(std::string fileName, optix::float4* data, int w, int h) {
  std::vector<float> channelR(w * h);
  std::vector<float> channelG(w * h);
  std::vector<float> channelB(w * h);
  for (int i = 0; i < w * h; ++i) {
    const optix::float4& px = data[i];
    channelR[i] = px.x / px.w;
    channelG[i] = px.y / px.w;
    channelB[i] = px.z / px.w;
  }

  EXRImage image;
  InitEXRImage(&image);

  const unsigned numChannels = 3;
  image.num_channels = numChannels;

  // Must be BGR(A) order, since most EXR viewers expect this channel order.
  const char* channel_names[] = { "B", "G", "R" };

  float* image_ptr[3] = {
    channelB.data(), // B
    channelG.data(), // G
    channelR.data()  // R
  };

  image.channel_names = channel_names;
  image.images = reinterpret_cast<unsigned char**>(image_ptr);
  image.width = w;
  image.height = h;
  image.compression = TINYEXR_COMPRESSIONTYPE_NONE;

  image.pixel_types = new int[sizeof(int) * numChannels];
  image.requested_pixel_types = new int[sizeof(int) * numChannels];
  for (int i = 0; i < image.num_channels; i++) {
    image.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
    image.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
  }

  const char* err;
  int ret = SaveMultiChannelEXRToFile(&image, fileName.c_str(), &err);

  delete[] image.pixel_types;
  delete[] image.requested_pixel_types;

  if (ret != 0) {
    throw std::runtime_error(
      str(format("Cannot read string property '%1%'") % std::string(err))
    );
  }
}

int main(int argc, char* argv[]) {
  using namespace boost::program_options;
 
  try {
    // Parse command-line args using boost::program_options.
    options_description desc("Allowed options");
    desc.add_options()
      ("help",
        "produce help message")
      ("input", value<std::string>()->required(),
        "JSON scene file input")
      ("output", value<std::string>()->default_value("output.exr"),
        "EXR output path");

    positional_options_description pd;
    pd.add("input", 1).add("output", 1);

    variables_map vars;
    store(
      command_line_parser(argc, argv).options(desc).positional(pd).run(), vars
    );

    // Print help message if requested by user.
    if (vars.count("help")) {
      std::cout << desc;
      return 0;
    }

    // Raise errors after checking the help flag.
    notify(vars);

    // Load scene and set up rendering.
    std::string input = vars["input"].as<std::string>();
    std::string output = vars["output"].as<std::string>();

    Scene scene(input);
    scene.defaultCamera()->prepare();

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
      "xray",
      SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      scene.defaultCamera()->pixelWidth(), scene.defaultCamera()->pixelHeight(),
      SDL_WINDOW_SHOWN
    );
    SDL_Surface* windowSurface = SDL_GetWindowSurface(window);
    
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point iterStartTime = startTime;
    bool down = false;
    int yOriginal;
    while (true) {
      int yOffset = 0;
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        if (event.type == SDL_MOUSEBUTTONDOWN) {
          down = true;
          yOriginal = event.button.y;
        } else if (event.type == SDL_MOUSEBUTTONUP) {
          down = false;
        } else if (event.type == SDL_MOUSEMOTION && down) {
          yOffset = event.motion.y - yOriginal;
        } else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE) {
          goto finish;
        }
      }

      // Move camera if the mouse-down offset > 1px.
      if (abs(yOffset) > 1) {
        yOriginal += yOffset;
        scene.defaultCamera()->translate(optix::make_float3(0, 0, yOffset * 0.25f));
      }

      // Render here!
      scene.defaultCamera()->render(!down);

      // Transfer current render to the screen.
      SDL_LockSurface(windowSurface);
      void* imageMapped = scene.defaultCamera()->imageBuffer()->map();
      std::memcpy(
        windowSurface->pixels,
        imageMapped,
        scene.defaultCamera()->pixelWidth() * scene.defaultCamera()->pixelHeight() * 4
      );
      scene.defaultCamera()->imageBuffer()->unmap();
      SDL_UnlockSurface(windowSurface);
      SDL_UpdateWindowSurface(window);

      // Update title with statistics.
      int frameNumber = scene.defaultCamera()->frameNumber();
      std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
      std::chrono::duration<float> iterRunTime =
        std::chrono::duration_cast<std::chrono::duration<float>>(endTime - iterStartTime);
      std::chrono::duration<float> totalRunTime =
        std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime);
      float fps = 1.0f / iterRunTime.count();
      float totalSecs = totalRunTime.count();
      iterStartTime = endTime;

      std::string title = str(format("xray [Iteration %1%, %2$.f fps, %3$.1f\" elapsed]") % frameNumber % fps % totalSecs);
      SDL_SetWindowTitle(window, title.c_str());
    }

finish:
    // Save the output before exiting.
    optix::float4* accumMapped = static_cast<optix::float4*>(scene.defaultCamera()->accumBuffer()->map());
    writeToEXR(output, accumMapped, scene.defaultCamera()->pixelWidth(), scene.defaultCamera()->pixelHeight());
    scene.defaultCamera()->accumBuffer()->unmap();
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    return 42;
  }

  return 0;
}
