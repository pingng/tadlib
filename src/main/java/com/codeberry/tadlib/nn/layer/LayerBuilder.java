package com.codeberry.tadlib.nn.layer;

import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.util.StringUtils;

import java.util.Random;

@StringUtils.OutputJsonClassValue
public interface LayerBuilder {
    Layer build(Random rnd, Shape inputShape);
}
