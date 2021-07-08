import matplotlib.pyplot as plt


epoch = []; loss = []; reconst_loss = []; kl_div_hidden = []; kl_div_cell=[]; bleu_score_avg = [];gaussian_score = []
tf_ratio = [];klw = []
with open("result_train.txt") as file:
    for line in file:
        values = line.split(" ")
        epoch.append(int(values[0]))
        loss.append(float(values[1]))
        reconst_loss.append(float(values[2]))
        kl_div_hidden.append(float(values[3]))
        kl_div_cell.append(float(values[4]))
        bleu_score_avg.append(float(values[5]))
        gaussian_score.append(float(values[6]))
        klw.append(float(values[7]))
        tf_ratio.append(float(values[8]))


# reconstion loss = Cross entropy (yellow line)
plt.plot(epoch, reconst_loss, color = 'orange')
plt.show()     

# hidden and cell KL divergence (blue and dark blue line)
fig, ax1 = plt.subplots()
ax1.set_ylabel('loss', color='royalblue')  
ax1.plot(epoch, kl_div_cell, color='royalblue')
ax1.plot(epoch, kl_div_hidden, color='navy')
ax1.tick_params(axis='y', labelcolor='royalblue')

ax2 = ax1.twinx() 
ax2.set_xlabel('epoch')
ax2.set_ylabel('reconst_loss', color='orange')
ax2.plot(epoch, reconst_loss, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

fig.tight_layout()
plt.show()

# Teacher forcing ratio (red line)
fig, ax1 = plt.subplots()
ax1.set_ylabel('loss', color='royalblue') 
ax1.plot(epoch, kl_div_cell, color='royalblue')
ax1.plot(epoch, kl_div_hidden, color='navy')
ax1.plot(epoch, reconst_loss, color='orange')
ax1.tick_params(axis='y', labelcolor='royalblue')

ax2 = ax1.twinx() 
ax2.set_xlabel('epoch')
ax2.set_ylabel('klw', color='green')
ax2.plot(epoch, klw, color='green')
ax2.tick_params(axis='y', labelcolor='green')

fig.tight_layout() 
plt.show()

# KL weights (green line)
fig, ax1 = plt.subplots()
ax1.set_ylabel('loss', color='royalblue') 
ax1.plot(epoch, kl_div_cell, color='royalblue')
ax1.plot(epoch, kl_div_hidden, color='navy')
ax1.plot(epoch, reconst_loss, color='orange')
ax1.tick_params(axis='y', labelcolor='royalblue')


ax2 = ax1.twinx() 
ax2.set_xlabel('epoch')
ax2.set_ylabel('teacher forcing ratio', color='red')
ax2.plot(epoch, tf_ratio, color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.tight_layout()  
plt.show()

# BLEU score of test data (black line)
fig, ax1 = plt.subplots()
ax1.set_ylabel('loss', color='royalblue') 
ax1.plot(epoch, kl_div_cell, color='royalblue')
ax1.plot(epoch, kl_div_hidden, color='navy')
ax1.plot(epoch, reconst_loss, color='orange')
ax1.tick_params(axis='y', labelcolor='royalblue')

ax2 = ax1.twinx() 
ax2.set_xlabel('epoch')
ax2.set_ylabel('bleu score', color='black')
ax2.plot(epoch, bleu_score_avg, color='black')
ax2.tick_params(axis='y', labelcolor='black')

fig.tight_layout()
plt.show()

# Gaussian score (grey line)
fig, ax1 = plt.subplots()
ax1.set_ylabel('loss', color='royalblue') 
ax1.plot(epoch, kl_div_cell, color='royalblue')
ax1.plot(epoch, kl_div_hidden, color='navy')
ax1.plot(epoch, reconst_loss, color='orange')
ax1.tick_params(axis='y', labelcolor='royalblue')

ax2 = ax1.twinx()  
ax2.set_xlabel('epoch')
ax2.set_ylabel('gaussian score', color='grey')
ax2.plot(epoch, gaussian_score, color='grey')
ax2.tick_params(axis='y', labelcolor='grey')

fig.tight_layout() 
plt.show()