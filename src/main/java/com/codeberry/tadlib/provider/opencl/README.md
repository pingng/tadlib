OpenCL support
===
TADLib will create OpenCL context for the first available Platform. It has only been tested with nvidia gpus.

The native OpenCL library is accessed using [JNA](https://github.com/java-native-access/jna). It is 
a great way to use native libs with pure Java code.

Device initialization in context
---
All available devices will be included in the created context. The system property
_tad.opencl.devices_ can be used to control the included devices. It is a comma separated
list. Each element can be:

- a text, which is a partial string from the device name
- an index, which is the device index

Available devices will be listed at startup.

Device selection for operations
---
Operations will by default run on the first device. You can select the target
device for operation by wrapping the code with the method `ProviderStore.onDevice`:
```java
// Enqueue the opencl kernel on the device with the partial name "Xp"
Tensor c = ProviderStore.onDevice("Xp", () -> Ops.add(a, b));
```
Such wrapper calls will not have any special effects for the java provider. The
code will just run as normal, ignoring the concept of devices.

Memory usage on the device/gpu
---
TADLib keeps it simple by storing all data on in OpenCL. It is up to the
OpenCL implementation to decide if the data will be written to the device or not.

The main takeaway is that TADLib does nothing fancy to juggle data between
the java and the OpenCL. Everything is stored in OpenCL, and managed by OpenCL.

Data is only fetch from OpenCL when your code need to print or read it.

Memory management
---
Like most native libraries, OpenCL requires that all resources must be manually 
released. TADLib do use the `java.lang.ref.Cleaner` to provide _auto release_
of the resources. But you will get best performance by releasing buffers as
soon as you are done with them.

Resources can be released by:
- TODO: to be written... \
  com.codeberry.tadlib.memorymanagement.DisposalRegister.modelIteration
  com.codeberry.tadlib.memorymanagement.DisposalRegister.disposeAllExceptReturnedValues
  com.codeberry.tadlib.memorymanagement.DisposalRegister.disposeAllExceptContainedReturnValues

Leak Detection
---
TO BE WRITTEN...

com.codeberry.tadlib.memorymanagement.LeakDetector.printOldObjectsAndIncreaseObjectAge
com.codeberry.tadlib.memorymanagement.AbstractDisposer.TRACK_GC_RELEASE