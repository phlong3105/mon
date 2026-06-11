from typing import List

from hafnia.dataset.hafnia_dataset_types import License

LICENSE_TYPES: List[License] = [
    License(
        name="Creative Commons: Attribution-NonCommercial-ShareAlike 2.0 Generic",
        name_short="CC BY-NC-SA 2.0",
        url="https://creativecommons.org/licenses/by-nc-sa/2.0/",
    ),
    License(
        name="Creative Commons: Attribution-NonCommercial 2.0 Generic",
        name_short="CC BY-NC 2.0",
        url="https://creativecommons.org/licenses/by-nc/2.0/",
    ),
    License(
        name="Creative Commons: Attribution-NonCommercial-NoDerivs 2.0 Generic",
        name_short="CC BY-NC-ND 2.0",
        url="https://creativecommons.org/licenses/by-nc-nd/2.0/",
    ),
    License(
        name="Creative Commons: Attribution 2.0 Generic",
        name_short="CC BY 2.0",
        url="https://creativecommons.org/licenses/by/2.0/",
    ),
    License(
        name="Creative Commons: Attribution-ShareAlike 2.0 Generic",
        name_short="CC BY-SA 2.0",
        url="https://creativecommons.org/licenses/by-sa/2.0/",
    ),
    License(
        name="Creative Commons: Attribution-NoDerivs 2.0 Generic",
        name_short="CC BY-ND 2.0",
        url="https://creativecommons.org/licenses/by-nd/2.0/",
    ),
    License(
        name="Flickr: No known copyright restrictions",
        name_short="Flickr",
        url="https://flickr.com/commons/usage/",
    ),
    License(
        name="United States Government Work",
        name_short="US Gov",
        url="http://www.usa.gov/copyright.shtml",
    ),
]


def get_license_by_url(url: str) -> License:
    for license in LICENSE_TYPES:
        # To handle http urls
        license_url = (license.url or "").replace("http://", "https://")
        url_https = url.replace("http://", "https://")
        if license_url == url_https:
            return license
    raise ValueError(f"License with URL '{url}' not found.")


def get_license_by_short_name(short_name: str) -> License:
    for license in LICENSE_TYPES:
        if license.name_short == short_name:
            return license
    raise ValueError(f"License with short name '{short_name}' not found.")
