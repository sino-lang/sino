use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::types::IntType;
use inkwell::values::IntValue;
use std::error::Error;
use std::io::{self, Write};
use std::iter::Peekable;
use std::str::Chars;
// 引入获取系统信息的常量
use std::env::consts::OS;

// 辅助函数：跳过迭代器中的所有空白字符（空格/制表符，通用兼容）
fn skip_whitespace(chars: &mut Peekable<Chars<'_>>) {
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
}

// 辅助函数：判断字符是否为有效计算器字符
fn is_valid_char(c: char) -> bool {
    c.is_ascii_digit() || c == '+' || c == '-' || c == '*' || c == '/' || c == '(' || c == ')'
}

// 辅助函数：将字符串首字母大写（兼容全小写输入）
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => s.to_string(),
    }
}

// 计算器核心结构体：仅绑定LLVM上下文，生命周期贯穿所有LLVM对象
struct Calculator<'ctx> {
    context: &'ctx Context,
    // IR临时值计数器，避免命名冲突
    tmp_counter: u32,
}

// JIT函数类型：无参数，返回i64，遵循C调用规范（JIT执行强制要求）
type JitCalcFunc = unsafe extern "C" fn() -> i64;

impl<'ctx> Calculator<'ctx> {
    // 初始化：绑定上下文+重置计数器
    fn new(context: &'ctx Context) -> Self {
        Self {
            context,
            tmp_counter: 0,
        }
    }

    // 生成唯一的IR临时值名称，解决命名冲突
    fn gen_tmp_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.tmp_counter);
        self.tmp_counter += 1;
        name
    }

    // 表达式解析：处理加减（低优先级），递归委托乘除解析
    fn parse_expression(
        &mut self,
        expr: &mut Peekable<Chars<'_>>,
        builder: &inkwell::builder::Builder<'ctx>,
        i64_type: IntType<'ctx>,
        zero_val: IntValue<'ctx>,
    ) -> Result<IntValue<'ctx>, Box<dyn Error>> {
        let mut value = self.parse_term(expr, builder, i64_type, zero_val)?;

        loop {
            skip_whitespace(expr);
            let Some(&op) = expr.peek() else {
                break;
            };

            match op {
                '+' | '-' => {
                    expr.next(); // 消耗操作符
                    let rhs = self.parse_term(expr, builder, i64_type, zero_val)?;
                    value = match op {
                        '+' => builder.build_int_add(value, rhs, &self.gen_tmp_name("add_tmp"))?,
                        '-' => builder.build_int_sub(value, rhs, &self.gen_tmp_name("sub_tmp"))?,
                        _ => unreachable!(),
                    };
                }
                _ => break,
            }
        }

        Ok(value)
    }

    // 项解析：处理乘除（中优先级），递归委托因子解析，含精准除零检查
    fn parse_term(
        &mut self,
        expr: &mut Peekable<Chars<'_>>,
        builder: &inkwell::builder::Builder<'ctx>,
        i64_type: IntType<'ctx>,
        zero_val: IntValue<'ctx>,
    ) -> Result<IntValue<'ctx>, Box<dyn Error>> {
        let mut value = self.parse_factor(expr, builder, i64_type, zero_val)?;

        loop {
            skip_whitespace(expr);
            let Some(&op) = expr.peek() else {
                break;
            };

            match op {
                '*' | '/' => {
                    expr.next(); // 消耗操作符
                    let rhs = self.parse_factor(expr, builder, i64_type, zero_val)?;
                    value = match op {
                        '*' => builder.build_int_mul(value, rhs, &self.gen_tmp_name("mul_tmp"))?,
                        '/' => {
                            // 除零检查：常量除零直接抛错
                            if rhs.is_const() && rhs == zero_val {
                                return Err("ZeroDivisionError: division by zero".into());
                            }
                            // 唯一名称生成除法IR，无符号除法适配正整数场景
                            builder.build_int_unsigned_div(
                                value,
                                rhs,
                                &self.gen_tmp_name("div_tmp"),
                            )?
                        }
                        _ => unreachable!(),
                    };
                }
                _ => break,
            }
        }
        Ok(value)
    }

    // 因子解析：处理整数、括号（最高优先级），支持多层嵌套+任意空格兼容
    fn parse_factor(
        &mut self,
        expr: &mut Peekable<Chars<'_>>,
        builder: &inkwell::builder::Builder<'ctx>,
        i64_type: IntType<'ctx>,
        zero_val: IntValue<'ctx>,
    ) -> Result<IntValue<'ctx>, Box<dyn Error>> {
        skip_whitespace(expr);

        // 表达式意外结束，抛出Python风格语法错误
        let Some(&c) = expr.peek() else {
            return Err("SyntaxError: unexpected end of expression".into());
        };

        let result =
            match c {
                // 括号表达式：递归解析内部，支持无限层嵌套+括号后空白跳过
                '(' => {
                    expr.next(); // 消耗左括号
                    let inner_value = self.parse_expression(expr, builder, i64_type, zero_val)?;
                    skip_whitespace(expr); // 跳过括号内和右括号间的空白

                    // 匹配右括号，无匹配则抛错
                    if expr.peek() != Some(&')') {
                        return Err("SyntaxError: missing closing parenthesis ')'".into());
                    }
                    expr.next(); // 消耗右括号
                    skip_whitespace(expr); // 跳过右括号后的空白
                    inner_value
                }
                // 整数解析：提取连续数字，生成LLVM i64常量值
                '0'..='9' => {
                    let mut num_chars = String::new();
                    // 手动循环消耗数字，确保迭代器正确推进
                    while let Some(&ch) = expr.peek() {
                        if ch.is_ascii_digit() {
                            num_chars.push(ch);
                            expr.next(); // 主动消耗数字字符，推进迭代器
                        } else {
                            break;
                        }
                    }
                    let num = num_chars.parse::<i64>().map_err(|_e| {
                        format!(
                            "ValueError: invalid literal for int() with base 10: '{}'",
                            num_chars
                        )
                    })?;
                    i64_type.const_int(num as u64, false)
                }
                // 非法字符：抛出Python风格语法错误
                _ => return Err(format!(
                    "SyntaxError: invalid character '{}' (only 0-9, +, -, *, /, () are allowed)",
                    c
                )
                .into()),
            };

        Ok(result)
    }

    // 核心运行方法：整合LLVM全流程（创建→IR生成→JIT编译→执行）
    fn run(&mut self, expr_str: &str) -> Result<i64, Box<dyn Error>> {
        // 1. 初始化LLVM核心组件
        let module = self.context.create_module("calculator");
        let builder = self.context.create_builder();
        let i64_type = self.context.i64_type();
        let zero_val = i64_type.const_int(0, false);
        // 创建JIT执行引擎
        let engine = module
            .create_jit_execution_engine(OptimizationLevel::Default)
            .map_err(|e| format!("RuntimeError: LLVM initialization failed: {}", e))?;

        // 2. 定义LLVM主函数
        let main_func_type = i64_type.fn_type(&[], false);
        let main_func = module.add_function("main", main_func_type, None);
        let entry_block = self.context.append_basic_block(main_func, "entry");
        builder.position_at_end(entry_block);

        // 3. 解析用户表达式，生成LLVM IR
        let mut expr = expr_str.trim().chars().peekable();
        let expr_value = self.parse_expression(&mut expr, &builder, i64_type, zero_val)?;

        // 残留字符检查
        skip_whitespace(&mut expr);
        if let Some(&remaining_char) = expr.peek() {
            if !is_valid_char(remaining_char) {
                return Err(format!(
                    "SyntaxError: invalid trailing character '{}'",
                    remaining_char
                )
                .into());
            } else {
                return Err(format!(
                    "SyntaxError: incomplete expression, trailing character '{}'",
                    remaining_char
                )
                .into());
            }
        }

        // 4. 生成return指令
        builder
            .build_return(Some(&expr_value))
            .map_err(|e| format!("RuntimeError: failed to generate return instruction: {}", e))?;

        // 验证函数IR的合法性
        if !main_func.verify(true) {
            return Err("RuntimeError: invalid LLVM IR generated".into());
        }

        // 5. JIT即时编译 + 执行
        let jit_func = unsafe {
            engine
                .get_function::<JitCalcFunc>("main")
                .map_err(|e| format!("RuntimeError: JIT compilation failed: {}", e))?
        };
        let result = unsafe { jit_func.call() };

        Ok(result)
    }
}

// 模仿Python终端的启动信息打印函数
fn print_startup_info() {
    // 从Cargo.toml中读取项目名称和版本（编译时注入）
    let pkg_name = env!("CARGO_PKG_NAME");
    let pkg_version = env!("CARGO_PKG_VERSION");
    // 获取运行系统（linux/windows/macos等）
    let os = OS;

    // 项目名称首字母大写
    let capitalized_pkg_name = capitalize_first(pkg_name);

    // 格式化输出：移除[rustc ...]，保持格式简洁
    let startup_msg = format!("{} {} on {}", capitalized_pkg_name, pkg_version, os);
    println!("{}", startup_msg);
}

// 主函数：模仿Python终端样式的交互式入口
fn main() -> Result<(), Box<dyn Error>> {
    // 第一步：打印启动信息，类似Python终端的版本提示
    print_startup_info();

    // 创建LLVM全局上下文
    let context = Context::create();
    let mut calc = Calculator::new(&context);

    let mut input = String::new();
    loop {
        // 模仿Python终端提示符 `>>>`
        print!(">>> ");
        io::stdout().flush()?;

        // 读取用户输入
        input.clear();
        let read_result = io::stdin().read_line(&mut input);

        // 处理输入异常（Ctrl+C/Ctrl+D）
        if read_result.is_err() {
            println!("\n");
            break;
        }

        let expr = input.trim();

        // 空输入直接跳过，保持提示符样式
        if expr.is_empty() {
            continue;
        }

        // 退出条件：模仿Python终端，输入exit/quit退出
        if expr.eq_ignore_ascii_case("exit") || expr.eq_ignore_ascii_case("quit") {
            break;
        }

        // 重置临时值计数器
        calc.tmp_counter = 0;

        // 执行计算并模仿Python终端输出样式
        match calc.run(expr) {
            // 正确结果：直接打印数值，无额外装饰（和Python REPL一致）
            Ok(res) => println!("{}", res),
            // 错误结果：直接打印错误信息，去掉多余图标和文字
            Err(e) => println!("{}", e),
        }
    }

    Ok(())
}
