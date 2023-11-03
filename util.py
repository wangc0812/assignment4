import time
import os,torch
def train(epoch,network, train_loader, optimizer, loss_f, log_interval):
  train_losses = []
  train_counter = []
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = loss_f(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')


def test(network, loss_f, test_loader, test_iter = -1):
  test_losses = []
  network.eval()
  test_loss = 0
  correct = 0
  counter = 0
  with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
      if test_iter == i:
        break
      counter += 1
      output = network(data)
      test_loss += loss_f(output, target)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= counter
  test_losses.append(test_loss)
  if test_iter == -1:
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

def time_model_evaluation(model,loss_f, test_loader):
  eval_start_time = time.time()
  test(model,loss_f, test_loader)
  eval_end_time = time.time()
  eval_duration_time = eval_end_time - eval_start_time
  # print(result)
  print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


def print_size_of_model(model):
  torch.save(model.state_dict(), "temp.p")
  print('Size (MB):', os.path.getsize("temp.p") / 1e6)
  os.remove('temp.p')


