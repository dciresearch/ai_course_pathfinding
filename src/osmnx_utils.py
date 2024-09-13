from tqdm.auto import tqdm
import json
from pathlib import Path
import matplotlib
from collections import Counter, defaultdict
from IPython import display
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, ImageMagickWriter
import matplotlib.pyplot as plt
from dataclasses import dataclass
import igraph as ig
import networkx as nx
import numpy as np
from typing import Union, List
import osmnx as ox
import pandas as pd


def convert_to_ig(G, weight_attr='length'):
    old_vids = list(G.nodes.keys())
    node_int_map = {v: k for k, v in enumerate(old_vids)}

    G_nx = nx.relabel_nodes(G, node_int_map)

    G_ig = ig.Graph(directed=True)
    G_ig.add_vertices(G_nx.nodes)
    G_ig.add_edges(G_nx.edges())
    G_ig.vs["osmid"] = old_vids
    G_ig.es[weight_attr] = list(nx.get_edge_attributes(G_nx, weight_attr).values())

    old_to_new_node_map = node_int_map
    return G_ig, old_to_new_node_map


def get_distance_matrix(G, target_nodes, weight_attr='length'):
    G_ig = G
    if isinstance(G_ig, nx.Graph):
        G_ig, old_to_new_node_map = convert_to_ig(G_ig, weight_attr=weight_attr)

        target_nodes = [old_to_new_node_map[node] for node in target_nodes]

    sp_ig = G_ig.distances(target_nodes, target_nodes, weights=weight_attr)
    sp_ig = np.array(sp_ig)
    return sp_ig


ROAD_SPEEDS = {"residential": 35, "secondary": 50, "tertiary": 60}


def add_travel_time(G):
    G = ox.add_edge_speeds(G, hwy_speeds=ROAD_SPEEDS)
    G = ox.add_edge_travel_times(G)


def get_nearest_nodes(target_locs, graph):
    x, y = list(zip(*((r.x, r.y) for r in target_locs.itertuples())))
    return ox.nearest_nodes(graph, x, y)


@dataclass
class TSPInstance:
    target_nodes: int
    target_descriptions: pd.DataFrame
    distance_matrix: np.ndarray
    shortest_paths: List[List[List[int]]]


class PlaceGraph:
    def __init__(self, place: Union[str, List[float]], road_mode='drive', cache_dir="./osmnx_cached") -> None:
        self._place = place
        if isinstance(self._place, str):
            self.nx_G = ox.graph_from_place(self._place, network_type=road_mode)
            self._feature_builder = ox.features_from_place
        else:
            self.nx_G = ox.graph_from_bbox(self._place, network_type=road_mode)
            self._feature_builder = ox.features_from_bbox

        add_travel_time(self.nx_G)

        self.G, self.old_to_new_node_map = convert_to_ig(self.nx_G, weight_attr='travel_time')
        self.new_to_old_map = {v: k for k, v in self.old_to_new_node_map.items()}
        self._traffic_edges = None
        self._traffic_groups = None
        self._cache_dir = Path(cache_dir)

    def get_locations_by_amenity(self, amenity: str, tag: str = 'amenity') -> pd.DataFrame:
        tags = {tag: amenity}
        locations = self._feature_builder(self._place, tags=tags)
        locations = locations.reset_index()
        if tag == 'building':
            locations['geometry'] = locations['geometry'].apply(lambda x: x.centroid)
        else:
            locations = locations[locations["element_type"] == "node"]
        return locations

    def convert_fast_node_indices_to_slow(self, fast_indices):
        return [self.new_to_old_map[i] for i in fast_indices]

    def get_respective_nodes(self, targets):
        target_locs = pd.DataFrame(
            [(row.osmid, row.geometry.x, row.geometry.y) for row in targets.itertuples()],
            columns=['osmid', 'x', 'y'])

        # nearest neighbor may give the same nodes
        target_nodes = list(set(get_nearest_nodes(target_locs, self.nx_G)))
        return target_nodes

    def get_travelling_salesman_problem(self, targets: Union[str, pd.DataFrame], ratio=0.5):
        targets = self.get_locations_by_amenity(targets)
        targets_index = np.random.choice(
            np.arange(targets.shape[0]),
            size=int(targets.shape[0]*ratio),
            replace=False
        )

        target_nodes = self.get_respective_nodes(targets.iloc[targets_index])
        ig_target_nodes = [self.old_to_new_node_map[n] for n in target_nodes]
        dist_mat = get_distance_matrix(self.G, ig_target_nodes, weight_attr='travel_time')

        # filter out unreachable nodes
        indices = np.arange(dist_mat.shape[0])
        bad_indices = []
        for i in range(2):
            mask = (~np.isfinite(dist_mat)).sum(axis=i) > dist_mat.shape[0]//2
            bad_indices.extend(indices[mask].tolist())
        good_indices = list(set(indices.tolist())-set(bad_indices))

        target_nodes = [target_nodes[i] for i in good_indices]
        ig_target_nodes = [ig_target_nodes[i] for i in good_indices]
        dist_mat = dist_mat[good_indices][:, good_indices]

        # record the paths
        paths = []
        for i in range(0, len(ig_target_nodes)):
            paths.append(self.G.get_shortest_paths(
                ig_target_nodes[i], ig_target_nodes,  weights='travel_time', output="vpath"))
        paths = [[[self.new_to_old_map[n] for n in path] for path in row] for row in paths]

        return TSPInstance(target_nodes, targets, dist_mat, paths)

    def get_node_coordinates(self, node_id: int):
        return (self.nx_G.nodes[node_id]['x'], self.nx_G.nodes[node_id]['y'])

    def _load_traffic_groups(self):
        if self._traffic_groups is None:
            with open(self._cache_dir / "traffic_groups.json") as f:
                groups = json.load(f)

            group_nodes = {}
            for gn, queries in tqdm(groups.items(), desc="Loading points of interest..."):
                group_nodes[gn] = []
                for tag, arguments in queries:
                    try:
                        results = self.get_locations_by_amenity(amenity=arguments.split(), tag=tag)
                    except ValueError:
                        continue
                    # print(tag, arguments, len(results))
                    if not results.empty:
                        nodes = self.get_respective_nodes(results)
                        group_nodes[gn].extend(nodes)
            self._traffic_groups = group_nodes
        return self._traffic_groups

    def _load_traffic_rules(self):
        with open(self._cache_dir / "traffic_rules.json") as f:
            rules = json.load(f)
        return rules

    def set_online_mode(self, steps=24, landmark_size=0.1, seed=1234, traffic_factor=1.0):
        self._update_freq = len(self.G.vs)//steps
        for e in self.G.es:
            e['traffic'] = 0.0
            e['real_travel_time'] = e['travel_time']

        # generate landmark dict
        self._ldmks = self._load_traffic_groups()

        total_ldmks = sum(map(len, self._ldmks.values()))

        rule_table = self._load_traffic_rules()

        # init first sources for step 0
        src_size = int(total_ldmks*landmark_size)
        flat_ldmks = [(k, l)
                      for k, v in self._ldmks.items() for l in v
                      ]

        rng = np.random.default_rng(seed)
        flat_ids = rng.choice(
            range(len(flat_ldmks)),
            size=src_size, replace=False
        )
        flat_ldmks = [flat_ldmks[i] for i in flat_ids]
        sources = defaultdict(list)
        for g, v in flat_ldmks:
            sources[g].append(v)

        self._step_pairs = []
        for tgt in range(steps):
            pairs = []
            next_sources = defaultdict(list)
            for src_group, src_lds in sources.items():
                tgt_ldmks, probs = list(
                    zip(*(((g, ld, rule_table[g]["impact"]), p)
                          for g, p in rule_table[src_group]['adjacent'].items() for ld in self._ldmks[g]))
                )
                probs = np.array(probs)
                probs /= probs.sum()
                tgt_ldmks_ids = rng.choice(range(len(tgt_ldmks)), len(src_lds), p=probs)
                tgt_ldmks = [tgt_ldmks[i] for i in tgt_ldmks_ids]
                for src, (g, tgt, impact) in zip(src_lds, tgt_ldmks):
                    pairs.append((src, tgt, impact * traffic_factor))
                    next_sources[g].append(tgt)
            self._step_pairs.append(pairs)
            sources = next_sources

    def _update_edge_real_speed(self):
        for e in self._traffic_edges:
            edge = self.G.es[e]
            edge['real_travel_time'] = edge['travel_time']*(0.2 + 0.8 * (1-edge['traffic']))

    def _schedule_traffic(self, step):
        cur_step = step % len(self._step_pairs)
        next_step = (step + 1) % len(self._step_pairs)

        edges_to_track = []
        for step in [cur_step, next_step]:
            step_edges = []
            for src, tgt, impact in tqdm(
                self._step_pairs[step], desc='loading traffic...', leave=False
            ):
                src = self.old_to_new_node_map[src]
                tgt = self.old_to_new_node_map[tgt]
                route = self.G.get_shortest_path(src, tgt, weights='real_travel_time')
                if len(route) == 0:
                    continue
                for a, b in zip(route, route[1:]):
                    e = self.G.get_eid(a, b)
                    step_edges.append(e)
                    edge = self.G.es[e]
                    edge['traffic'] += impact

            is_active = (step == cur_step)
            for e in step_edges:
                edge = self.G.es[e]
                edge['traffic'] = np.clip(edge['traffic'], 0, 1)
                if not is_active:
                    edge['traffic'] /= self._update_freq
                edge['steps_left'] = self._update_freq * (2*is_active - 1)
            edges_to_track.extend(step_edges)
        self._traffic_edges = edges_to_track
        self._update_edge_real_speed()

    def start_traffic(self):
        self._update_step = 0
        self._schedule_traffic(0)

    def get_online_update(self):
        """
        Updates graph speeds in accordance to traffic situation
        """
        out_of_updates = False
        for e in self._traffic_edges:
            edge = self.G.es[e]
            sign = np.sign(edge['steps_left'])
            out_of_updates = sign == 0
            if out_of_updates:
                break
            factor = self._update_freq*(sign < 0) + edge['steps_left']
            if factor:
                edge['traffic'] -= edge['traffic']/factor * sign
                edge['steps_left'] -= sign

        self._update_edge_real_speed()

        if out_of_updates:
            self._update_step += 1
            self._schedule_traffic(self._update_step)

    def plot_graph(self, targets=None, **kwargs):
        node_c = 'w'
        node_s = 1
        node_ec = 'none'
        if isinstance(targets, list):
            node_c, node_s, node_ec = list(
                zip(*(('w', 1, 'none') if n not in targets else ('orange', 24, 'r') for n in self.nx_G.nodes)))
        return ox.plot_graph(
            self.nx_G, node_color=node_c, node_size=node_s,
            node_edgecolor=node_ec, bgcolor='#373737', **kwargs
        )

    def plot_route(self, route: List[int], targets=[], nodes_are_fast: bool = False):
        if nodes_are_fast:
            route = self.convert_fast_node_indices_to_slow(route)

        node_c, node_s, node_ec = ('w', 1, 'none')
        if isinstance(targets, list):
            node_c, node_s, node_ec = list(
                zip(*(('w', 1, 'none') if n not in targets else ('orange', 24, 'r') for n in self.nx_G.nodes)))

        ox.plot_graph_route(
            self.nx_G, route, route_linewidth=2, node_color=node_c,
            orig_dest_size=10, route_color='g', bgcolor='#373737',
            node_size=node_s, node_edgecolor=node_ec, route_alpha=0.9
        )

    def get_aspects(self):
        x, y = zip(*((x['x'], x['y']) for v in self.G.vs if (x := v['name'])))
        h = max(y) - min(y)
        w = max(x) - min(x)
        h, w = 2*h/(h+w), 2*w/(h+w)
        return w, h

    def plot_route_animated(self, route: List[int], targets=[],
                            nodes_are_fast: bool = False, online_mode=False, render_every=None):
        if nodes_are_fast:
            route = self.convert_fast_node_indices_to_slow(route)
            targets = self.convert_fast_node_indices_to_slow(targets)

        node_c, node_s, node_ec = ('w', 1, 'none')
        if isinstance(targets, list):
            node_c, node_s, node_ec = list(
                zip(*(('w', 1, 'none') if n not in targets else ('orange', 34, 'r') for n in self.nx_G.nodes)))

        fig, ax = ox.plot_graph(
            self.nx_G, show=False, close=False, bgcolor='#373737',
            node_color=node_c, node_size=node_s, node_edgecolor=node_ec
        )
        # print(type(ax), dir(ax.collections[0]))

        w, h = self.get_aspects()

        fig.set_size_inches(8*w, 8*h, forward=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

        node_XY = {node: (data['x'], data['y']) for node, data in self.nx_G.nodes(data=True)}
        x, y = zip(*[node_XY[node] for node in route])

        line, = ax.plot([], [], 'g-', linewidth=3, alpha=0.9)
        point, = ax.plot([], [], 'go', markersize=5)

        if render_every is None:
            render_every = int(np.ceil(len(x)/50))

        total_frames = int(np.ceil(len(x)/render_every))+5

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate(i):
            i = i*render_every
            if online_mode:
                edge_line_collection = ax.collections[0]
                dc = "#999999"
                cmap = matplotlib.colormaps["YlOrRd"]
                ec = [
                    dc if e not in self._traffic_edges else cmap(0.1+0.9*self.G.es[e]['traffic']) for e in range(len(self.nx_G.edges))
                ]
                edge_line_collection.set_color(ec)
                self.get_online_update()
            line.set_data(x[:i], y[:i])
            i = min(i, len(x) - 1)
            point.set_data([x[i]], [y[i]])
            return line, point

        if online_mode:
            self.start_traffic()

        fig.tight_layout()
        print('Animating...')
        ani = FuncAnimation(fig, animate, frames=total_frames, init_func=init,
                            interval=200, blit=True, repeat=False, repeat_delay=3)
        print('Converting...')
        # TODO speed up the animation saving process
        ani.save('tmp.gif',
                 writer=FFMpegWriter(fps=10, extra_args=["-sws_flags", "fast_bilinear"])
                 )
        plt.close()
        print('Rendering...')
        return display.Image(filename='tmp.gif')
