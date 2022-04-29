# graduation_experiment



reference_code: colight参考代码

resources：所有资源文件，包括roadnet file， flow file等

my_code： 我编写的简单代码





engine：

```python
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['E:\graduation\lib\*', 'E:\graduation\SEUCityflow-1.0.0.jar'],
                   convertStrings=True)  # jvmargs 根据实际数据设定
Engine = jpype.JClass('engine')
engine = Engine("E:\graduation\cityflow_config_file.json", 2)
engine.next_step()
print(engine.get_vehicle_count())
```