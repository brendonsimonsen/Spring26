CS6610 Program 1

lexer.rs --------------

**Overview**

The lexer takes raw assembly source code as text and converts it into structured tokens that the parser can understand. These tokens are a way of grouping or breaking the lines of assembly code into structured parts. Instead of working with strings, the parser receives meaningful units like labels, directives, instructions, and data declarations.

**tokenize()**

This function loops through every line and decides what kind of token it represents. Each line of code is tested against known patterns: directives, labels, data declarations, and instructions. When a match is found, the appropriate token is created.

**Helper functions**
extract\_comment - splits code from comments using the hash character.

parse\_directive - recognizes .text and .data directives.

parse\_lable\_with\_rest - validates label names and supports labels appearing on the same line as instructions.

parse\_word\_declaration - parses .word declarations, including repeat syntax like -1:16.

parse\_instruction - splits a line into mnemonic and operands.

parse\_number - handles decimal and hexadecimal numbers.



**Conclusion**

Overall, the lexer converts raw text into structured token that make the parser's job much simpler and more reliable.



parser.rs -------------

**Overview**

The parser takes the tokens produced by the lexer and converts them into a structured program representation. It tracks which section we’re in, builds a symbol table for labels, calculates memory addresses, and organizes instructions and data so they can be encoded later.

**(for loop with match)**

The for loop iterates through every token and reacts based on its type. It builds a ParsedProgram while keeping track of labels that appear before instructions or data

**(TOKEN TYPES)**

**Directives** - Directives change the current section. If we see .text or .data, we update the parser state so future instructions or data are handled correctly.

**Labels** - Labels are stored temporarily. They get attached to the next instruction or data declaration so we can assign them an address.

**Instruction** - When an instruction is encountered, the parser first verifies that it’s in the .text section. It calculates how many real instructions it expands into, handles any pending label, builds an Instruction struct, and updates the program counter.

**Word declaration** - For declarations, the parser ensures it’s in the .data section, assigns labels an address in data memory, creates a DataWord structure, and updates the data offset.

**Conclusion** - At the end, the parser returns a fully structured program containing instructions, data, and labels, which is then passed to the encoder to generate binary output.





encoder.rs ----------------

**Overview**

The encoder takes the structured program produced by the parser and converts it into a binary string representing machine code. It calculates section sizes, encodes each instruction into 32-bit binary, expands pseudo-instructions, and appends all data values.

**Encode()**

The encode function builds the final output in four parts: text size, data size, encoded instructions, and encoded data.

**Text\_size**

We count how many real instructions exist, including expanded pseudo-instructions, and multiply by 4 bytes per instruction.

**data\_size**

Data size is calculated by counting how many words exist and multiplying by 4 bytes.

Both sizes are encoded as 32-bit binary values at the beginning of the output.

**encode\_instruction**

This function will select the correct encoding function based on the mnemonic.

I will only walk through two of the functions I created in more detail to save time.

**R-type instruction**

R-type instructions follow a fixed binary format: opcode, registers, shift amount, and function code.

Registers are parsed from strings into numeric register IDs.

The function code determines the exact operation.

Bit shifting packs all fields into a single 32-bit machine instruction.

**Psuedo-Instruction Expansion example (encode\_la)**

Psuedo-instructions don't exist in hardware, so they expand into real instructions.

The label's address is split into upper and lower 16 bits.

Two instructions are generated and concatenated into the output.

One assembly instruction becomes two machine instructions.

**Helper functions**

The helper functions handle parsing registers, parsing immediates, calculating branch offsets, and mapping opcodes and function codes. This keeps the main encoding logic clean and readable.

**Conclusion**

Overall, the encoder transforms structured instructions into precise binary machine code while handling instruction formats, pseudo-instruction expansion, and memory layout.



Run example --------------------

