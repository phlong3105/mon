//
// Created by Long H. Pham on 1/3/24.
//

#include "path.hpp"
#include "../global.hpp"

#pragma region Path
namespace mon {
    
    auto is_file(const fs::path& path, const fs::file_status& status) -> bool {
        if (fs::status_known(status) ? fs::exists(status) : fs::exists(path))
            return true;
        return false;
    }

    auto is_image_file(const fs::path& path, const bool exist) -> bool  {
        const auto ext = path.extension().string();
        auto exts = ImageFormat::values;
        return (exist ? is_file(path) : true) & (std::find(exts.begin(), exts.end(), ext) != exts.end());
    }

    auto get_image_file(const fs::path& path) -> fs::path {
        auto parent = path.parent_path();
        for (const auto& ext: ImageFormat::values) {
            auto temp = path.parent_path() / path.stem();
            temp += ext;
            if (is_file(temp)) {
                return temp;
            }
        }
        return path;
    }
    
}
#pragma endregion Path
