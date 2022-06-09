    def backward(self, dout):
    """
    Returns
    ----------
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
      - dW: Gradient with respect to W, of shape (D, M)
      - db: Gradient with respect to b, of shape (M,)
        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0) # To support multiple data (batch)
        # Find the derivative of the bias by the sum of the previous derivatives
        
        return dx
