package com.codeberry.tadlib.provider.opencl;

import com.sun.jna.Function;
import com.sun.jna.Function.PostCallRead;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByReference;
import com.sun.jna.ptr.PointerByReference;

public class OclEventByReference extends PointerByReference {
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

    public void oclRelease() {
        OclEvent ev = getValue();
        if (ev != null) {
            ev.oclRelease();
            setValue(null);
        }
    }
}
