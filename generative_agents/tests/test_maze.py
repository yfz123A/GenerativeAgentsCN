"""L1 单元测试：modules/maze.py 迷宫与寻路（合成迷你迷宫，不依赖真实 2MB 地图）。"""

import pytest

from modules import utils
from modules.maze import Maze

ADDRESS_KEYS = ["world", "sector", "arena", "game_object"]


def make_maze(wall_x=None, gap_ys=(2,), room=None, size=6):
    """构造 6x6 合成迷宫：可选一列墙（可留缺口）+ 一个带地址的房间。"""
    tiles = []
    for y in range(size):
        if wall_x is not None and y not in gap_ys:
            tiles.append({"coord": [wall_x, y], "collision": True})
    if room:
        tiles.append({"coord": list(room), "address": ["小屋", "床", "床单"]})
    config = {
        "size": [size, size],
        "tile_size": 32,
        "world": "the Ville",
        "tile_address_keys": ADDRESS_KEYS,
        "tiles": tiles,
    }
    return Maze(config, utils.create_io_logger("warn"))


def _adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# ============ 加载与地址 ============

def test_maze_loads_with_empty_tiles_by_default():
    maze = make_maze()
    tile = maze.tile_at([0, 1])
    assert list(tile.coord) == [0, 1]  # 产品里 coord 以 tuple 存储
    assert tile.is_empty
    assert tile.get_address(as_list=True) == ["the Ville"]


def test_addressed_room_tile():
    maze = make_maze(room=[1, 1])
    tile = maze.tile_at([1, 1])
    assert not tile.is_empty
    assert tile.get_address(as_list=True) == ["the Ville", "小屋", "床", "床单"]
    assert tile.get_address("arena", as_list=False) == "the Ville:小屋:床"
    assert tile.has_address("game_object")
    # 4 级地址自动登记物件事件
    assert any(e.subject == "床单" for e in tile.get_events())


def test_address_tiles_lookup():
    maze = make_maze(room=[1, 1])
    assert maze.get_address_tiles(["the Ville", "小屋"]) == {(1, 1)}


def test_collision_tile_exists():
    maze = make_maze(wall_x=3, gap_ys=(2,))
    assert maze.tile_at([3, 0]).collision
    assert not maze.tile_at([3, 2]).collision  # 缺口


# ============ 寻路 ============

def test_find_path_adjacent():
    maze = make_maze()
    # 已知行为：返回路径包含起点（tuple），终点保持传入类型（list）
    path = [list(c) for c in maze.find_path([1, 1], [2, 1])]
    assert path == [[1, 1], [2, 1]]


def test_find_path_routes_through_gap():
    maze = make_maze(wall_x=3, gap_ys=(2,))
    path = [list(c) for c in maze.find_path([1, 1], [4, 4])]
    assert path, "应当能穿过缺口"
    assert path[0] == [1, 1] and path[-1] == [4, 4]
    for a, b in zip(path, path[1:]):
        assert _adjacent(a, b)
    assert [3, 2] in path  # 必经缺口


def test_find_path_unreachable_returns_empty():
    maze = make_maze(wall_x=3, gap_ys=())  # 整列墙，左右不通
    assert maze.find_path([1, 1], [4, 4]) == []


def test_find_path_border_destination_is_unreachable():
    # 已知行为：BFS 只在 0 < x < width-1 的内部格扩展，边界格永远到不了
    maze = make_maze()
    assert maze.find_path([1, 1], [0, 0]) == []


# ============ 感知范围与邻格 ============

def test_get_scope_box_mode():
    maze = make_maze()
    tiles = maze.get_scope([2, 2], {"mode": "box", "vision_r": 1})
    coords = {tuple(t.coord) for t in tiles}
    assert coords == {(x, y) for x in (1, 2, 3) for y in (1, 2, 3)}


def test_get_around_filters_collision():
    maze = make_maze(wall_x=3, gap_ys=(2,))
    # (3,1) 与 (3,3) 是墙，no_collision=True 时被过滤
    around = {tuple(c) for c in maze.get_around([3, 2])}
    assert around == {(2, 2), (4, 2)}
    around_blocked = {tuple(c) for c in maze.get_around([3, 1])}
    assert (3, 2) in around_blocked and (3, 0) not in around_blocked


# ============ 物件事件更新 ============

def test_update_obj_updates_all_tiles_of_address():
    maze = make_maze(room=[1, 1])
    from modules.memory.event import Event

    addr = ["the Ville", "小屋", "床", "床单"]
    event = Event("床单", "被", "李四", address=addr)
    maze.update_obj([1, 1], event)
    tile = maze.tile_at([1, 1])
    assert any(e.subject == "床单" and e.object == "李四" for e in tile.get_events())
