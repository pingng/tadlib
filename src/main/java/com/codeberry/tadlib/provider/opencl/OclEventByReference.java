package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.provider.opencl.jna.TADPointerByReference;
import com.sun.jna.Pointer;

public class OclEventByReference extends TADPointerByReference {
    private OclEvent oclEvent;

    public OclEventByReference() {
        super();
    }

    public OclEvent getValue() {
        return oclEvent;
    }

    public void setValue(OclEvent event) {
        this.oclEvent = event;
        super.setValue(event);
    }

    public void onFinishedCall() {
        Pointer evPointer = super.getValue();
        if (evPointer == null) {
            oclEvent = null;
        } else {
            this.oclEvent = OclEvent.createEvent(Pointer.nativeValue(evPointer));
        }
    }

    @Override
    public void dispose() {
        OclEvent ev = getValue();
        if (ev != null) {
            ev.oclRelease();
            setValue(null);
        }
        super.dispose();
    }

    @Override
    protected Class<?> getDataType() {
        return OclEvent.class;
    }

    @Override
    protected String getStrValue() {
        return String.valueOf(getValue());
    }
}
