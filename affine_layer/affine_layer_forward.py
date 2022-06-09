    def forward(self, x):
    """
    Parameters
    ----------
          - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
          - W: A numpy array of weights, of shape (D, M)
          - b: A numpy array of biases, of shape (M,)
        """
        self.x = x
        out = np.dot(x, self.W) + self.b
        
        return out
