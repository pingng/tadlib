Single Value Auto Grad
===
This package is a standalone implementation of automatic gradient.
It only supports single scalar values, and is easier to understand.

Example
---

### Simple usage

For variable *a* and *b*, find the values so that **(a + b) == 11** and **(a * b) == 18**:

```java
Value a = newParam(rndZeroMean(rand));
Value b = newParam(rndZeroMean(rand));

for (int epoch = 0; epoch < EPOCHS; epoch++) {
    // sumLoss = ((a + b) - 11)^2
    Value sumLoss = sqr(sub(add(a, b), constant(11)));

    // productLoss = ((a * b) - 18)^2
    Value productLoss = sqr(sub(mul(a, b), constant(18)));

    Value loss = add(sumLoss, productLoss);

    loss.backward();

    // Update params
    a = value(a.v - a.grad * LEARNING_RATE);
    b = value(b.v - b.grad * LEARNING_RATE);
}
```

See the [example source](example/SimpleExample.java).

### Solving an Age Problem

Let us try to solve this assignment:

```
Adam is 24 years older than Belle, but in six years,
Adam will by three times older than Belle.
  Q: How old is Adam?
```

The answer can be found by minimizing the loss of the following equations:

```
From the equation: 
    Adam - 24 = Belle
...minimize (to zero):
    firstLoss = (Adam - 24 - Belle)^2

From the equation:
    (Adam + 6) = (Belle + 6) * 3.
...minimize (to zero):
    secondLoss = ((Adam + 6) - (Belle + 6) * 3)^2
```

Code:

```java
Value adam = newParam(rndZeroMean(rand));
Value belle = newParam(rndZeroMean(rand));

for (int epoch = 0; epoch < EPOCHS; epoch++) {
    // Adam - 24 = Belle
    Value firstLeft = sub(adam, constant(24));
    Value firstRight = belle;
    Value firstCriteria = sub(firstLeft, firstRight);
    Value firstLoss = sqr(firstCriteria);

    // (Adam + 6) = (Belle + 6) * 3.
    Value secondLeft = add(adam, constant(6));
    Value secondRight = mul(add(belle, constant(6)), constant(3));
    Value secondCriteria = sub(secondLeft, secondRight);
    Value secondLoss = sqr(secondCriteria);

    Value loss = add(firstLoss, secondLoss);

    loss.backward();

    // Update params
    adam = value(adam.v - adam.grad * LEARNING_RATE);
    belle = value(belle.v - belle.grad * LEARNING_RATE);
}
```

See the [example source](example/AgeProblemExample.java).