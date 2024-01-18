//
// Created by Long H. Pham on 1/3/24.
//

#ifndef PATH_HPP
#define PATH_HPP

#include <filesystem>
namespace fs = std::filesystem;

#pragma region Path
namespace mon {
    
    auto is_file(const fs::path& path, const fs::file_status& status = fs::file_status{}) -> bool;
    auto is_image_file(const fs::path& path, bool exist = true) -> bool;
    auto get_image_file(const fs::path& path) -> fs::path;
    
}
#pragma endregion Path

#endif //PATH_HPP
