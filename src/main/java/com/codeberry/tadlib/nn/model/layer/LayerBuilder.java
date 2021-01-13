package com.codeberry.tadlib.nn.model.layer;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.util.StringUtils;

import java.util.Random;

@StringUtils.OutputJsonClassValue
public interface LayerBuilder {
    Layer build(Random rnd, Shape inputShape);
}
