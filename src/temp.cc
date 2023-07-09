#include "../include/Utils.hpp"




int main() {
    bool _cudaSupported  = false;


    // Obtain information from the OpenCV compilation
    // Here is a lot of information.
    const cv::String str = cv::getBuildInformation();

    // Parse looking for "Use Cuda" or the option you are looking for.
    std::istringstream strStream(str);

    std::string line;
    while (std::getline(strStream, line))
    {
        // Enable this to see all the options. (Remember to remove the break)
        std::cout << line << std::endl;

        // if(line.find("Use Cuda") != std::string::npos)
        // {
        //     // Trim from elft.
        //     line.erase(line.begin(), std::find_if(line.begin(), line.end(),
        //     std::not1(std::ptr_fun<int, int>(std::isspace))));

        //     // Trim from right.
        //     line.erase(line.begin(), std::find_if(line.begin(), line.end(),
        //     std::not1(std::ptr_fun<int, int>(std::isspace))));

        //     // Convert to lowercase may not be necessary.
        //     std::transform(line.begin(), line.end(), line.begin(), ::tolower);
        //     if (line.find("yes") != std::string::npos)
        //     {
        //         std::cout << "USE CUDA = YES" << std::endl;
        //         _cudaSupported = true;
        //         break;
        //     }
        // }
    }
}