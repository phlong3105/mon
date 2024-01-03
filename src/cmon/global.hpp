//
// Created by Long H. Pham on 1/3/24.
//

#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include <string>
#include <stdexcept>
#include <filesystem>
namespace fs = std::filesystem;


#pragma region Directory
namespace mon {

    const fs::path _current_path = fs::current_path();
    const fs::path ROOT_DIR      = _current_path.parent_path();
    const fs::path SOURCE_DIR    = ROOT_DIR / "src";
    const fs::path BIN_DIR       = ROOT_DIR / "bin";
    const fs::path DATA_DIR      = ROOT_DIR / "data";
    const fs::path DOCS_DIR      = ROOT_DIR / "docs";
    const fs::path RUN_DIR       = ROOT_DIR / "run";
    const fs::path TEST_DIR      = ROOT_DIR / "test";
    
}
#pragma endregion Directory


#pragma region Enum
namespace mon {

    class ImageFormat {
    public:
        enum ImageFormatEnum {
            arw  = 0,
            bmp  = 1,
            dng  = 2,
            jpg  = 3,
            jpeg = 4,
            png  = 5,
            ppm  = 6,
            raf  = 7,
            tif  = 8,
            tiff = 9,
        };

        inline static const std::vector<std::string> values {
            ".arw",
            ".bmp",
            ".dng",
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".raf",
            ".tif",
            ".tiff"
        };
        
        explicit ImageFormat(const ImageFormatEnum image_format) { value=image_format; };

        explicit ImageFormat(const int int_value) {
            if (int_value >= values.size()) {
                value = static_cast<ImageFormatEnum>(int_value);
            }
            else {
                throw std::invalid_argument( "Wrong value" );
            }
        };

        explicit ImageFormat(const std::string& string_value) {
            if (const auto it = std::find(values.begin(), values.end(), string_value); it != values.end()) {
                value = static_cast<ImageFormatEnum>(it - values.begin());
            }
            else {
                throw std::invalid_argument( "Wrong value" );
            }
        };

    private:
        ImageFormatEnum value;
    };
    
}
#pragma endregion Enum

#endif //GLOBAL_HPP
