package com.codeberry.tadlib.provider;

import com.codeberry.tadlib.array.NDArray;

public class ProviderStore {
    private static Provider provider;

    public static void setProvider(Provider provider) {
        ProviderStore.provider = provider;
    }

    public static NDArray array(double v) {
        return provider.createArray(v);
    }
}
