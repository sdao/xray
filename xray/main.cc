#include <iostream>
#include <SDL.h>
#include <boost/program_options.hpp>
#include <chrono>
#include "scene.h"
#include "camera.h"

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

    SDL_Init(SDL_INIT_VIDEO);
    scene.defaultCamera()->prepare();

    SDL_Window* window = SDL_CreateWindow(
      "xray",
      SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      scene.defaultCamera()->pixelWidth(), scene.defaultCamera()->pixelHeight(),
      SDL_WINDOW_SHOWN
    );
    SDL_Surface* windowSurface = SDL_GetWindowSurface(window);
    
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point iterStartTime = startTime;
    while (true) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) {}

      scene.defaultCamera()->render();

      void* imageMapped = scene.defaultCamera()->imageBuffer()->map();
      SDL_LockSurface(windowSurface);
      std::memcpy(
        windowSurface->pixels,
        imageMapped,
        scene.defaultCamera()->pixelWidth() * scene.defaultCamera()->pixelHeight() * 4
      );
      SDL_UnlockSurface(windowSurface);
      scene.defaultCamera()->imageBuffer()->unmap();
      
      SDL_UpdateWindowSurface(window);

      std::cout << ".";
      int frameNumber = scene.defaultCamera()->frameNumber();
      if (frameNumber % 50 == 0) {
        std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();
        std::chrono::duration<float> iterRunTime =
          std::chrono::duration_cast<std::chrono::duration<float>>(endTime - iterStartTime);
        std::chrono::duration<float> totalRunTime =
          std::chrono::duration_cast<std::chrono::duration<float>>(endTime - startTime);
        float fps = 50.0f / iterRunTime.count();
        float totalSecs = totalRunTime.count();
        std::cout << " " << frameNumber << " (" << fps << " fps, " << totalSecs << "s elapsed)" << std::endl;
        iterStartTime = endTime;
      }
    }
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    return 42;
  }

  return 0;
}
