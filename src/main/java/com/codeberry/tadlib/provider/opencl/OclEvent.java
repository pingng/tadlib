package com.codeberry.tadlib.provider.opencl;

import com.codeberry.tadlib.memorymanagement.AbstractDisposer;
import com.codeberry.tadlib.memorymanagement.DisposalRegister;
import com.sun.jna.Pointer;

import static com.codeberry.tadlib.provider.opencl.consts.ErrorCode.throwOnError;
import static com.sun.jna.Pointer.createConstant;

public class OclEvent extends Pointer {
    private final Disposer disposer;

    private static class Disposer extends AbstractDisposer {
        private final long peer;
        private boolean released;

        public Disposer(long peer) {
            this.peer = peer;
        }

        @Override
        protected void releaseResource() {
            OpenCL.INSTANCE.clReleaseEvent(createConstant(peer));
        }

        @Override
        protected long getResourceId() {
            return peer;
        }
    }

    private OclEvent(long peer) {
        super(peer);

        this.disposer = new Disposer(peer);
    }

    public static OclEvent createEvent(long peer) {
        OclEvent ev = new OclEvent(peer);
        DisposalRegister.registerDisposer(ev, ev.disposer);
        return ev;
    }

    public void oclRelease() {
        this.disposer.release();
    }
}
