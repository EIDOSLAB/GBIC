from torch_geometric.data import Data

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t

def flat_nodes(x,shape):
  B,C,W,H = shape
  #x = x.reshape((-1,C,H*W)).contiguous()
  x = x.transpose(1,2)# .contiguous()
  x = x.reshape((B*H*W,C))# .contiguous()
  return x


def unflat_nodes(x,shape):
  B,C,W,H = shape

  x = x.reshape((B,H*W,C))
  x = x.transpose(1,2)
  x = x.reshape((-1,C,H,W))
  return x