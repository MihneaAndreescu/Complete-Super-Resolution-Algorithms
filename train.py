from torch.utils.data import DataLoader
import torch

def train_loop(dataloader, model, loss_fn, optimizer):
    train_loss = 0
    
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss /= len(dataloader)
    return train_loss


def train(index, model, epochs, training_data, batch_size, learning_rate, loss_function, name): # functia returneaza modelul
    mn = 1e9
    timeSinceMn = 0
    
    train_dataloader = DataLoader(training_data, batch_size = batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

    
    for epc in range(1, epochs + 1):
        '''if keyboard.is_pressed('w'):  # if key 'q' is pressed 
            learning_rate *= 2
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 
        
        if keyboard.is_pressed('s'):
            learning_rate /= 2
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)'''
 
        
        train_loss = train_loop(train_dataloader, model, loss_function, optimizer)
        if train_loss < mn:
            mn = train_loss 
            timeSinceMn = 0
            torch.save(model.state_dict(), name)
        else:
            timeSinceMn += 1
         
        print("(Adam) ----->", epc, "out of", epochs, "loss =", "{:.9f}".format(train_loss), "| learning rate =", learning_rate, "time since min =", timeSinceMn)

    return model