# Notes

## self attention vs cross attention

self: K, Q, V come from the same source X

cross:
Q from X

K, V from diff src

used when there is a separate src of nodes and we'd like to pull info into our set of nodes

## scaling factor

if Q.K are matmul'ed naively, the variance of the result will be on the order of the head size

the result should be diffused to avoid peak attention to a single node in context

## positional encoding

[Refer to article](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
