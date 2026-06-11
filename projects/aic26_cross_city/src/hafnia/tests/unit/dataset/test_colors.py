from hafnia.dataset.colors import get_n_colors


def test_get_n_colors():
    ten_colors = get_n_colors(10)
    assert len(ten_colors) == 10
    assert len(get_n_colors(100)) == 100
    assert len(get_n_colors(1000)) == 1000
