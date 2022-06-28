import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
import json
import copy
from collections import defaultdict


class PFNDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.feature_names = {
                "node": ['APF', 'BASE_KV', 'BS', 'BUS_AREA', 'COST1', 'COST2',
                         'COST3', 'GEN_STATUS', 'GS', 'LAM_P', 'LAM_Q', 'MBASE',
                         'MODEL', 'MU_PMAX', 'MU_PMIN', 'MU_QMAX', 'MU_QMIN', 'MU_VMAX',
                         'MU_VMIN', 'NCOST', 'PC1', 'PC2', 'PD', 'PG', 'PMAX', 'PMIN', 'QC1MAX',
                         'QC1MIN', 'QC2MAX', 'QC2MIN', 'QD', 'QG', 'QMAX', 'QMIN', 'RAMP_10',
                         'RAMP_30', 'RAMP_AGC', 'RAMP_Q', 'SHUTDOWN', 'STARTUP', 'TYPE', 'VA',
                         'VG', 'VM', 'VMAX', 'VMIN', 'ZONE'],
                "edge": ['ANGMAX', 'ANGMIN', 'BR_B', 'BR_R', 'BR_STATUS', 'BR_X', 'MU_ANGMAX',
                         'MU_ANGMIN', 'MU_SF', 'MU_ST', 'PF', 'PT', 'QF', 'QT', 'RATE_A',
                         'RATE_B', 'RATE_C', 'SHIFT', 'TAP', 'IN_ORIGIN']
        }
        self.feature_sizes = {k: len(v) for k, v in self.feature_names.items()}
        self.processed_path = Path(f"{root}/processed")
        self.raw_path = Path(f"{root}/raw")
        self._raw_file_names = [str(p) for p in self.raw_path.iterdir()]
        self._process_file_names = [str(self.processed_path.joinpath(f"{Path(p).stem}.pt")) for p in self._raw_file_names]
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return self._process_file_names

    def download(self):
        pass

    def _extract_nodes_features(self, node_dict):
        return [node_dict[k] if k in node_dict.keys() else 0 for k in self.feature_names["node"]]

    def _extract_edges_features(self, edge_dict):
        return [edge_dict[k] if k in edge_dict.keys() else 0 for k in self.feature_names["edge"][:-1]]

    def _get_idx_nodes(self, nodes_features):
        return sorted([int(i) for i in nodes_features.keys()])

    def _get_nodes_idx(self, nodes_features, idx_nodes):
        return {v: i for i, v in enumerate(idx_nodes)}

    def _get_edge_tuple(self, rep):
        rep = rep.replace('[', '')
        rep = rep.replace(']', '')
        rep = list(map(int, rep.split(',')))
        return rep[0], rep[1]

    def _get_edge_str(self, edge, add_one=False):
        if add_one:
            edge[0] += 1
            edge[1] += 1
        return f"[{edge[0]}, {edge[1]}]"

    def raw_to_data(self, json_data):
        # Nodes
        print("ICI")
        idx_nodes = self._get_idx_nodes(json_data["nodes_features"])

        nv = len(json_data["nodes_features"])

        ## Nodes features matrix
        y = json_data["solving_time"]
        x_nodes = torch.zeros((nv, self.feature_sizes["node"]))
        for i in range(nv):
            x_nodes[i, :] = torch.tensor(self._extract_nodes_features(json_data["nodes_features"][str(idx_nodes[i])]))

        # Edges
        ne = len(json_data["edges_features"])

        ## Graph connectivity
        edges = [[e[0] - 1, e[1] - 1] for e in json_data["all_edges"]]
        e1 = [e[0] for e in edges]
        e2 = [e[1] for e in edges]
        edge_index = torch.tensor([e1 + e2, e2 + e1], dtype=torch.long)

        ## Edges features matrix
        x_edges = torch.zeros((ne, self.feature_sizes["edge"]))
        for i in range(ne):
            idx = self._get_edge_str(edges[i], add_one=False)
            if idx in json_data["edges_features"].keys():
                x_edges[i, :] = torch.tensor(self._extract_edges_features(json_data["edges_features"][idx]) + [1])
            else:
                x_edges[i, :] = torch.zeros(self.feature_sizes["edge"])
        x_edges = torch.cat((x_edges, x_edges), 0)


        return Data(x_nodes, edge_index, y=y, edge_attr=x_edges)

    def build_instance_to_idx(self):
        self.instance_to_idx = defaultdict(list)
        for i, data in enumerate(self.meta_data):
            self.instance_to_idx[data["origin_name"]].append(i)

    def process(self):
        self.meta_data = []
        for i, raw_path in enumerate(self.raw_file_names):
            json_data = json.load(open(raw_path, "r"))
            self.meta_data.append({"uuid": json_data["uuid"], "origin_name": json_data["origin_name"], "origin_scenario": json_data["origin_scenario"]})
            if not self.processed_path.joinpath(f"{Path(raw_path).stem}.pt").is_file():
                data = self.raw_to_data(json_data)
                torch.save(data, self.processed_path.joinpath(f"{Path(raw_path).stem}.pt"))
        self.build_instance_to_idx()

    def _keep_indexes(self, indexes):
        self.meta_data = [m for i, m in enumerate(self.meta_data) if i in indexes]
        self._raw_file_names = [f for i, f in enumerate(self._raw_file_names) if i in indexes]
        self._process_file_names = [f for i, f in enumerate(self._process_file_names) if i in indexes]
        self.build_instance_to_idx()

    def exclude(self, exclude_instance):
        excluded_dataset = copy.deepcopy(self)
        excluded_indexes = []

        # Retrieve the indexes to exclude
        for instance in exclude_instance:
            excluded_indexes.extend(self.instance_to_idx[instance])
        keep_indexes = [i for i in range(len(self)) if i not in excluded_indexes]

        # Split
        self._keep_indexes(keep_indexes)
        excluded_dataset._keep_indexes(excluded_indexes)

        return excluded_dataset

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        return data


def display_dataset(dataset):
    print(len(dataset))
    print(list(dataset.instance_to_idx.keys()))
    for k, v in dataset.instance_to_idx.items():
        print(f"{k} = {v}")
    for i, data in enumerate(dataset.meta_data):
        print(f"{i} = {data['origin_name']}")
