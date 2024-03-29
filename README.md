# ADDDSim2Real

Aduiduidui's repo for Sim2Real competition.

啦啦啦！

This is the `bailan` result of the so-called `ADuiDuiDui` team's result.

No additional package is needed to run the agent for a properly built `sim2real` environment.

耶耶耶！

## New method

**Run main program**

- Standard run：`python elwin_main.py --anime_run --no_graphics`

```bash
Usage: elwin_main.py [-h] [--anime_plan] [--anime_curv] [--anime_dyob]
                     [--anime_run] [--no_graphics]
Optional arguments:
  -h, --help     show this help message and exit
  --anime_plan   show animation when planning
  --anime_curv   show curvature of a path
  --anime_dyob   show dynamic obstacles
  --anime_run    show real time running animation
  --no_graphics  don't show simulator graphics
```

**Run modules in folder *elwin***

- Path planning & tracking test: `python planner.py --anime_plan --anime_curv `
- Re-generate the cost map: `python costmap.py `

## Stage_2 信息

### 相同的地方：

requirment list没有进行修改，之前的list还可以继续用

### 一些不同的地方：

1. 在规则的第八页更新了机器人平台更详细参数
2. P9更新了执行过程（仿真器在此根据P9的描述进行了修改）
   * 增加了机器人动力学参数接口（摩擦力参数、机器人重量、每个轮子的转动惯量、电机参数（电流力矩常数）和 PID 参数），可以在仿真环境 reset时指定参数值
   * 在观测中增加了速度状态反馈，具体见 api_test.py
   * .......(详细信息见规则的第九页)
3. P10列出了一些仿真到实体迁移可能需要考虑的问题
4. 更新了一个“submit_test”的文件
5. 模拟器更新到了3.0（但是没有Mac的？？）

### 第二阶段的测试细节：

6 月 20 日--7 月 30 日 Sim2Real 阶段，成功晋级（最低要求：测试过程中平均每次实验激活目标点数量不少于 3 个，在此基础上根据得分择优选择）的参赛队伍需要根据组委会反馈的实体测试结果对控制算法进行优化，提交过程如下：

1. 参赛队伍将代码提交至比赛官网；

2. 组委会下载（6 月 26 日之后的每周一）各参赛队伍更新的代码并部署到EP 机器人上执行任务。在测试过程中，考虑到机器人的安全和硬件损耗问题，工作人员首先会用简单测试等级对算法进行评估，当算法平均碰撞时间小于 20 秒时，才进行实际系统测试。如果实际系统测试过程中累积碰撞时间超过 20 秒时，则会人为干预强制终止程序。当机器人在测试过程中出现冲撞障碍物行为，工作人员会强制终止测试并向参赛队伍反馈测试问题。

3. 测试后，将 EP 机器人的数据和比赛录像反馈给参赛队伍；

4. 参赛队伍根据反馈数据和比赛录像进行算法优化。

## References

#### Open Source Code

https://github.com/ArianJM/rapidly-exploring-random-trees
