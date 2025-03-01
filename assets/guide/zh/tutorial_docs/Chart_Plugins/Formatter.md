# 格式化插件

VChart 从 `v1.10.0` 开始，支持了格式化扩展插件，提供更多格式化字符串的能力，包括：

1. 数值文本格式化
2. 时间文本格式化
3. 数据变量替换

## 如何使用格式化插件

格式化字符串支持使用数据内容替换，变量名需要用单括号`{}`括起来，例如：

```js
// 数据
{
  year: 2016,
  population: 899447
},

formatter: `"The population in {year} year is {population}"`。
```

### 数值数据

数值格式化采用和 [d3-formatter](https://d3js.org/d3-format) 类似的格式化约定。在格式化表达式中，与变量名之间用冒号隔开，例如：

- 小数点后两位： `{value:.2f}`
- 千位分隔符，无小数位： `{value:,.0f}`
- 千位分隔符，小数点后一位： `{value:,.1f}`

常用的格式化类型有：

- `e`：指数。
- `f`：浮点数。
- `g`：十进制或指数，四舍五入到有效数字。
- `r`：十进制，四舍五入到有效数字。
- `s`：带有 SI prefix（SI 前缀）的十进制数，四舍五入为有效数字。
- `%`：乘以 100，带有百分号的十进制数。
- `p`：乘以 100，四舍五入为有效数字，带有百分号的十进制数。
- `b`：二进制数，四舍五入为整数。
- `o`：八进制数，四舍五入为整数。
- `d`：十进制数，四舍五入为整数。
- `x`：十六进制数，使用小写字母，四舍五入为整数。
- `X`：十六进制数，使用大写字母，四舍五入为整数。
- `c`：在打印之前将整数转换为对应的 unicode 字符。
- ​`\(空\) - 和g`类似，但是会去掉尾随零

除此以外，格式化插件还扩展了两种常用的数值格式化逻辑：

- `t`： 保留指定小数位数，但不进行四舍五入。
- `z`：小数数值，四舍五入到有效数字；整数数值，无小数部分。

例如：

```js
// 数据
{
  value: 12.3893333;
  value2: 100;
}

formatter: '{value:.2f}'; // "12.39"
formatter: '{value:.2t}'; //  "12.38"

formatter: '{value:.2z}'; //  "12.39"
formatter: '{value2:.2z}'; //  "100"
```

`~` 选项会在所有格式类型中修剪不重要的尾零。最常用的是与类型 r 、 e 、 s 和 % 结合使用。例如：

```js
// 数据
{
  value: 1500;
}

formatter: '{value:s}'; // "1.50000k"
formatter: '{value:~s}'; //  "1.5k"
```

### 时间数据

与数值一样，日期也允许在冒号后面添加格式。允许的格式约定与 [d3-time-format](https://d3js.org/d3-time-format) 类似。例如：

```js
// 数据
{
  date: +new Date(2024, 5, 1);
}

// 完整日期: %Y-%m-%d
formatter: '{date:%Y-%m-%d}'; // "2024-05-01"
```

常用的日期格式化配置：

| **日期粒度** | **格式配置** | **日期内容**         | **示例**  | **数值范围** |
| ------------ | ------------ | -------------------- | --------- | ------------ |
| 年           | %Y           | 年的全称             | 2022      |              |
| 月           | %b           | 简写的月             | Jul       |              |
|              | %B           | 月的全称             | July      |              |
|              | %m           | 月份                 | 7         | [01, 12]     |
| 周           | %a           | 简写的周             | Wed       |              |
|              | %A           | 周的全称             | Wednesday |              |
| 日           | %d           | 使用 0 填补位数的天  | 1         | [01, 31]     |
|              | %e           | 使用空格填补位数的天 | 1         | [ 1, 31]     |
| 时           | %H           | 24 小时制小时        | 1         | [00, 23]     |
|              | %I           | 12 小时制小时        | 1         | [01, 12]     |
|              | %p           | AM 或 PM             | AM        |              |
| 分           | %M           | 分钟                 | 0         | [00, 59]     |
| 秒           | %S           | 秒                   | 0         | [00, 61]     |
| 毫秒         | %L           | 毫秒                 | 1         | [000, 999]   |

## 按需加载如何引入格式化插件

在通过按需加载使用 VChart 时，格式化插件需要手动注册使用：

```js
import { VChart } from '@visactor/vchart';
import { registerFormatPlugin } from '@visactor/vchart';

VChart.useRegisters([registerFormatPlugin]);
```
