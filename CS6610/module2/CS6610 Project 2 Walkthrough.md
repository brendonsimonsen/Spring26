CS6610 Project 2 Walkthrough



-----------------------------------------

**Decode.rs**

-----------------------------------------

This file implements the decode stage of a single-cycle MIPS simulator.

It will take the raw instruction from project 1, decode it, read the needed registers, and prepare values for the execute stage.



First we get the instruction.

This converts it into a structured Instruction type and determines whether it is R-type, I-type, or J-type.



Next, using a match statement, I read the register values for the appropriate instruction type.

For both R-type and I-type instructions, I read rs and rt using cpu.get\_reg.

For J-type instructions, no registers are needed, so I just return zeros.



This next part will handle a sign extension for I-type instructions.

(I cast the 16-bit immediate to an i32, which automatically performs sign extension in Rust.)



Finally, I package the information and return it to be sent to the execute stage.



To summarize, this stage converts a raw instruction into structured data that the rest of the pipeline can use.



-----------------------------------------------

**Execute.rs**

-----------------------------------------------

The execute stage is where the behavior of the instruction happens.

This is where the simulator will perform arithmetic operations, logical operations, branch decisions, and where jump targets are computed.



Here I initialize the variables that will be calculated based on the instruction. 



Then the decoded instruction is matched to its instruction format. For the R- and I-Type instructions, a helper function is called to execute the instruction. Then the jump logic is handled.

For J-Type instructions, the jump target is computed. For the JAL instruction, the return address is stored in 'alu\_result'.



execute\_r\_type:

Match on the funct field, then perform the appropriate operation. 



\*\*Important note: For signed comparisons and arithmetic shifts, we first need to cast to i32 to get correct signed behavior.



execute\_i\_type:

This will handle the immediate ALU instructions, branch instructions, and load and store instructions.



First are immediate ALU instructions. These directly compute a result using the sign-extended immediate.



Second are branch instructions.



I check the branch condition inside the match and store it as a Boolean variable. Then I compute the branch\_target after the match statement. 



Third are load and store instructions.



At the end of the execute stage, I return all the values needed by the memory stage.



------------------------------------------------

**Memory.rs**

------------------------------------------------

The memory stage is going to handle the data memory access and finalize which register will be written back, if applicable.



To begin, I match the instruction to its appropriate format, R-, I-, or J-Type.



R-Type:
For R-type instructions, I only need to determine whether the instruction writes back to a register.



Most R-type instructions write to rd.



However, I explicitly disable write-back for control and system instructions like JR and SYSCALL.



JALR still writes to rd, so it is handled by the default case.



I-Type:
For I-type instructions, match on the opcode.



For load instructions, I use the address that was computed in the execute stage.



For byte and half-word loads:

align the address to a word,

read the full word from memory,

extract the correct byte or halfword using shifts,

and then either sign-extend or zero-extend the value.

The result is written back to the rt register.



For store instructions, I again use the address from the execute stage.



For byte and half-word stores, I read the existing word, modify only the targeted portion using masks and shifts, and then write the updated word back.



Store instructions do not write to any register.



J-Type
For J-type instructions, only JAL writes a return address to register $31.



A normal J instruction has no write-back.



I return the selected destination register and the data to write as a MemWbData structure, which is then used by the write-back stage.



----------------------------

**simulator.rs**

----------------------------

For the simulator we pretty much follow this number list provided. And then be sure to increment on number 4. That was the issue I had emailed you about. I forgot to increment. 



-----------------------------

Run tests. 



