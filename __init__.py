import unittest

import torch
from torch.autograd import Variable, Function
import knn_pytorch


class KNearestNeighbor(Function):
  """ Compute k nearest neighbors for each query point.
  """
  def __init__(self, k):
    self.k = k

  def forward(self, ref, query):
    ref = ref.float().cuda()
    query = query.float().cuda()

    inds = torch.empty(query.shape[0], self.k, query.shape[2]).long().cuda()

    knn_pytorch.knn(ref, query, inds)

    return inds


class TestKNearestNeighbor(unittest.TestCase):

  def test_forward(self):
    D, N, M = 128, 100, 1000
    ref = Variable(torch.rand(2, D, N))
    query = Variable(torch.rand(2, D, M))

    inds = KNearestNeighbor(2)(ref, query)
    print inds


if __name__ == '__main__':
  unittest.main()
