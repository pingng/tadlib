package com.codeberry.tadlib.nn.model.layer;

import com.codeberry.tadlib.array.Shape;

import java.util.Random;

public interface LayerBuilder {
    Layer build(Random rnd, Shape inputShape);
}
