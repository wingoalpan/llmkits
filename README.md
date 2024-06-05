# llmkits -- Some useful helper functions for LLM models analysis

## 1. nnutils -- utils for neutral network module (pytorch)

currently, this module provide two functions: `get_net_detail(...)` and `check_net(...)`. 

- <strong> get_net_detail(...) </strong>
  <p>this function get the parameters count and detail model structures,include:</p>
  <p>a. the parent-children relationships of each layer </p>
  <p>b. weight/bias shape of each layer </p>
  <p>c. parameters count of each layer </p>
  <p>d. total parameter count of this model</p>
<p>

- <strong> check_net(...) </strong>
  <p>this function check if each layer's parameters shape (output shape) matches its forward layer's input shape. This only applicable to simple forward pattern.</p>
