/*
使用js实现一个mini版本的宏定义解析及展开，需要支持：
- 首先找到所有形如`printf(格式化字符串,变量列表);`的宏定义 
  和C语言的printf定义是一模一样
- printf可以多行；格式化字符串里面存在转义字符，只需要支持\t \n 2种最简单的情况即可；只支持%d；变量可以是数组中的值
- 函数返回值：
  返回所有的开始和结束位置信息列表
    每一个找到的printf的格式化字符串的拆分以及对应的变量名的所有列表

举例：定义输入字符串为：
const template = `
    printf("x=%dy>%d z=%d|m=%d",
            x,y[ 0] ,
            z[1],     mx);            
`;
输出
[{"startPos": 5,
  "endPos": 82,
  "parts": [["x=","x"],
            ["y>","y[ 0]"],
            [" z=","z[1]"],
            ["|m=","mx"]]}]
*/
function parsePrintfMacro(template) {
    const results = [];
    // 使用更宽松的正则表达式来匹配跨行的printf
    const printfRegex = /printf\s*\(\s*"([\s\S]*?)"\s*,\s*([\s\S]*?)\s*\)\s*;/g;

    let match;
    while ((match = printfRegex.exec(template)) !== null) {
        const startPos = match.index;
        const endPos = match.index + match[0].length;

        let formatStr = match[1];
        const variablesStr = match[2];

        // 处理续行：移除行末的反斜杠和换行符
        formatStr = formatStr.replace(/\\\s*\n\s*/g, '');

        // 正确拆分格式字符串：只按 %d 分隔
        const formatParts = [];
        let currentPart = '';
        let i = 0;
        while (i < formatStr.length) {
            if (formatStr[i] === '\\') {
                // 处理转义字符（只支持\t和\n）
                if (i + 1 < formatStr.length) {
                    const nextChar = formatStr[i + 1];
                    if (nextChar === 't' || nextChar === 'n') {
                        currentPart += formatStr[i] + formatStr[i + 1];
                        i += 2;
                    } else {
                        // 其他转义字符当作普通字符处理
                        currentPart += formatStr[i] + formatStr[i + 1];
                        i += 2;
                    }
                } else {
                    currentPart += formatStr[i];
                    i++;
                }
            } else if (formatStr[i] === '%' && i + 1 < formatStr.length && formatStr[i + 1] === 'd') {
                formatParts.push(currentPart);
                currentPart = '';
                i += 2;
            } else {
                currentPart += formatStr[i];
                i++;
            }
        }
        formatParts.push(currentPart);

        // 变量部分：去除多余空格和换行，再按逗号分隔
        const cleanedVariablesStr = variablesStr.replace(/\s+/g, ' ').trim();
        const variables = cleanedVariablesStr.split(/\s*,\s*/).map(v => v.trim());

        // 格式部分数量应该比变量数多1（因为是split('%d')）
        if (formatParts.length !== variables.length + 1) {
            throw new Error(
                `printf format parts count (${formatParts.length - 1}) doesn't match variables count (${variables.length})\n` +
                `Format string: "${formatStr}"\n` +
                `Format parts: ${JSON.stringify(formatParts)}\n` +
                `Variables string: "${variablesStr}"\n` +
                `Variables: ${JSON.stringify(variables)}\n` +
                `Full match: "${match[0]}"`
            );
        }

        // 组合格式和变量：第 i 个变量对应 formatParts[i] 和 variables[i]
        const parts = variables.map((varName, i) => [formatParts[i], varName]);

        results.push({
            startPos,
            endPos,
            parts
        });
    }

    return results;
}


/*
完成解析得到结构化的printf后，需要展开：
- 首先对于原来的printf所在区域的所有行抠出来，相当于切割出来后，前面加//注释。 
- 然后对于每一个part：
    字符串依次展开为c('a');c('b'); 然后变量展开为i(x); 单独一行

外层会定义webgpu的debug函数：
fn c(c: u32, thread_id: u32) {} // 写入字符（ASCII码）
fn i(num: i32, thread_id: u32) {} // 写入整数（支持负数）

举例：定义输入字符串为：
const template = `
    int x=1;
    int y= 2; printf("x=%dy>%d z=%d|m=%d",
            x,y[ 0] ,
            z[1],     mx); float othVar=3.0             
`;
输出
    int x=1;
    int y= 2;
    //printf("x=%dy>%d z=%d|m=%d",
    //        x,y[ 0] ,
    //        z[1],     mx);
c('x',t);c('=',t);i(x,t);
c('y',t);c('>',t);i(y[ 0],t);
c(' ',t);c('z',t);c('=',t);i(z[1],t);
c('|',t);c('m',t);c('=',t);i(mx,t);
    
    float othVar=3.0        
注意：非printf的内容要保留。
*/

function expandPrintfMacros(template) {
    const printfInfos = parsePrintfMacro(template);
    if (printfInfos.length === 0) {
        return template; // 没有 printf 宏，直接返回原代码
    }
    let result = '';
    let lastPos = 0;
    for (const printfInfo of printfInfos) {
        // 添加 printf 之前的内容
        result += template.slice(lastPos, printfInfo.startPos);
        // 获取原始printf代码（包括所有换行）
        const originalPrintf = template.slice(printfInfo.startPos, printfInfo.endPos);
        // 注释掉原始printf的每一行
        const commentedPrintf = originalPrintf.split('\n')
            .map(line => line.trim() ? `//${line}` : line) // 保留空行
            .join('\n');
        // 添加注释掉的printf
        result += commentedPrintf;
        // 确保注释和展开代码之间有换行
        if (!result.endsWith('\n')) {
            result += '\n';
        }

        // 重新解析格式字符串和变量来正确构建所有部分
        const match = /printf\s*\(\s*"([\s\S]*?)"\s*,\s*([\s\S]*?)\s*\)\s*;/g.exec(originalPrintf);
        if (match) {
            let formatStr = match[1];
            const variablesStr = match[2];

            // 处理续行：移除行末的反斜杠和换行符
            formatStr = formatStr.replace(/\\\s*\n\s*/g, '');

            // 正确拆分格式字符串：只按 %d 分隔
            const formatParts = [];
            let currentPart = '';
            let i = 0;
            while (i < formatStr.length) {
                if (formatStr[i] === '\\') {
                    // 处理转义字符（只支持\t和\n）
                    if (i + 1 < formatStr.length) {
                        const nextChar = formatStr[i + 1];
                        if (nextChar === 't' || nextChar === 'n') {
                            currentPart += formatStr[i] + formatStr[i + 1];
                            i += 2;
                        } else {
                            // 其他转义字符当作普通字符处理
                            currentPart += formatStr[i] + formatStr[i + 1];
                            i += 2;
                        }
                    } else {
                        currentPart += formatStr[i];
                        i++;
                    }
                } else if (formatStr[i] === '%' && i + 1 < formatStr.length && formatStr[i + 1] === 'd') {
                    formatParts.push(currentPart);
                    currentPart = '';
                    i += 2;
                } else {
                    currentPart += formatStr[i];
                    i++;
                }
            }
            formatParts.push(currentPart);

            // 变量部分：去除多余空格和换行，再按逗号分隔
            const cleanedVariablesStr = variablesStr.replace(/\s+/g, ' ').trim();
            const variables = cleanedVariablesStr.split(/\s*,\s*/).map(v => v.trim());

            // 展开printf宏
            let expandedCode = '';

            // 对于n个%d，应该有n+1个格式字符串部分
            for (let i = 0; i < formatParts.length; i++) {
                const formatPart = formatParts[i];
                // 处理格式字符串部分 - 转换为ASCII码
                let j = 0;
                while (j < formatPart.length) {
                    if (formatPart[j] === '\\') {
                        // 处理转义字符（只支持\t和\n）
                        const escapedChar = formatPart[j + 1];
                        let asciiCode;
                        switch (escapedChar) {
                            case 'n':
                                asciiCode = 10; // \n
                                break;
                            case 't':
                                asciiCode = 9;  // \t
                                break;
                            default:
                                // 其他转义字符当作普通字符处理
                                asciiCode = escapedChar.charCodeAt(0);
                        }
                        expandedCode += `c(${asciiCode},t);`;
                        j += 2;
                    } else {
                        const asciiCode = formatPart[j].charCodeAt(0);
                        expandedCode += `c(${asciiCode},w,t);`;
                        j++;
                    }
                }

                // 处理变量部分（如果不是最后一个格式字符串部分）
                if (i < variables.length) {
                    expandedCode += `i(${variables[i]},w,t);`;
                }
                expandedCode += '\n'; // 每个部分换行
            }

            // 添加展开后的代码
            result += expandedCode;
        }

        lastPos = printfInfo.endPos;
    }
    // 添加最后一个 printf 之后的内容
    result += template.slice(lastPos);
    return result;
}


