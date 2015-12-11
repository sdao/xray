#include "scene.h"
#include "camera.h"
#include "optix_helpers.h"
#include <iostream>
#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  using namespace boost::program_options;

  sUtilWrapper* sUtil;
  try {
    sUtil = new sUtilWrapper();
  } catch (char *err) {
    cout << err;
    return 1;
  }

  sUtil->initGlut(&argc, argv);

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
    scene.defaultCamera()->render();
    sUtil->DisplayBufferInGlutWindow(scene.getContext()->get(), argv[0], scene.defaultCamera()->getImageBuffer()->get());
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    return 42;
  }

  return 0;
}
