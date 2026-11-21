import mlx.core as mx


def loss_fn(w):
    return w * w


loss_and_grad = mx.value_and_grad(loss_fn)

for learning_rate in (0.1, 1.1):
    w = mx.array(3.0)
    print(f"\n学习率 {learning_rate}，初始 loss {loss_fn(w).item():.2f}")

    for step in range(1, 6):
        _, grad = loss_and_grad(w)
        w = w - learning_rate * grad
        print(f"第 {step} 步  w = {w.item():.2f}  loss = {loss_fn(w).item():.2f}")