using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using MegaLCSTest.OpenCL;

namespace MegaLCSTest;

// dotnet run -c Release
class Program{
    static void Main(string[] args){
        Console.WriteLine("Hello, World!");
        Summary summary1 = BenchmarkRunner.Run<Perf_HostLCSShared>();
    }
}