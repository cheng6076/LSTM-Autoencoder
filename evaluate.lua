require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'rnn'
require 'util.misc'
require 'util.Maskh'
require 'util.MaskedLoss'
cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-model','model checkpoint to use')
cmd:option('-data', 'dataset/test', 'dataset to use')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)

local protos = checkpoint.protos
local opt2 = checkpoint.opt
local idx2word, word2idx = checkpoint.vocab[1], checkpoint.vocab[2]

-- compute sentence embeddings
function eval(x)
    if opt.gpuid >= 0 then x = x:float():cuda() end
    m = x:clone():fill(1)
    local seq_length = x:size(2)
    local init_state = {}
    for L=1,opt2.num_layers do
      local h_init = torch.zeros(1, opt2.rnn_size)
      if opt.gpuid >=0 then h_init = h_init:cuda() end
      table.insert(init_state, h_init:clone())
      table.insert(init_state, h_init:clone())
    end
    local enc_state = {[0] = clone_list(init_state)}
    local lst
    for t=1,seq_length do
      protos.enc:evaluate()
      lst = protos.enc:forward{x[{{},t}], m[{{},t}], unpack(enc_state[t-1])}
      enc_state[t] = {}
      for i=1,#init_state do table.insert(enc_state[t], lst[i]) end
    end
    return enc_state[seq_length][#init_state]
end

f = io.open(opt.data, 'r')
result = io.open(opt.data .. 'emb', 'w')
for line in f:lines() do
  local sentence = {}
  table.insert(sentence, word2idx['START'])
  for word in line:gmatch'([^%s]+)' do
    if word2idx[word]~=nil then table.insert(sentence, word2idx[word]) end
  end
  local sentence_emb = torch.zeros(opt2.rnn_size)
  if #sentence>2 then
    sentence = torch.Tensor(sentence)
    sentence = sentence:view(1, -1)
    sentence_emb:copy(eval(sentence))
  end
  for i=1,sentence_emb:size(1) do
    result:write(sentence_emb[i])
    result:write(' ')
  end
  result:write('\n')
end
result:close()
f:close()
print ('Done!')
