
# A preview version submitted with paper review process. 

Requirements: tensorflow 2.2 +

Please report bugs via reviews.

#  Hints

- Implementation logic of the algorithm in the paper (also see comments in  CLPM/CLPM.py/_CLPM)
- alternation in CLPM_MLM.py: line 123

## Steps and some tricks.

	a. using UNIVERSAL/tokenizer to generate BPE and the entries of languages' domain. **NOTE that for your convience, you can use your FastBPE to collect language domains. e.g.,apply codes to corpus:./fast applybpe after-bpe-En-corpuse before-bpe-EN-corpuse bpe-codes and get ./fast getvocab after-bpe-En-corpuse > vocab.En)
	
	b. set up the entries of languages' domains for CLPM.  (CLPM.py, line 66)

	c. Select a masking method and a MLM instance.

	d. Suppose we use a XLM encoder and its masking method. The input is [x1, [MASK], x3,[MASK],x5,[MASK],x6,[MASK]]. In line 90 of CLPM_MLM.py, we random select [C] position: clpm_position [0,1,0,0,0,0,0,1]. We wrap the inference mode in line 105 of CLPM_MLM.py and use the inference mode in line 152 of CLPM.py. We pass the input [x1, [MASK], x3,[MASK],x5,[MASK],x6,[MASK]] for inferring. Then, we use the last hiden state to compute candidats (line line 144 def_cos of CLPM.py) and get [C]: [0,[C2],0,0,0,0,0,[C7]]. Please see comments.
	e. now we can get the perturbed input for training.
	Input * (1-  clpm_position) + [C] * clpm_position.
	e.g., [x1, [MASK], x3,[MASK],x5,[MASK],x6,[MASK]] * (1-[0,1,0,0,0,0,0,1]) =  [x1, 0, x3,[MASK],x5,[MASK],x6,0]. [x1, 0, x3,[MASK],x5,[MASK],x6,0] + [0,[C2],0,0,0,0,0,[C7]] = [x1, [C2], x3,[MASK],x5,[MASK],x6,[C7]]. [x1, [C2], x3,[MASK],x5,[MASK],x6,[C7]] is the training sample.



  	
	