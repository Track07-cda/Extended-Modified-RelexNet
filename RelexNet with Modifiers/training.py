import torch.nn.functional as F


def train_model(dataset, model, optimizer, scheduler, num_epochs, dev):
    losses = []
    for epoch in range(num_epochs):
        # training mode
        # dataset.set_partition(dataset.train)
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        count = 0
        for x, y, meta, meta_wi in dataset.train_dataset:
            # for every batch in the training dataset perform one update step of the optimizer.
            state = None
            model.zero_grad()
            y_h, state = model(x.to(dev), meta, meta_wi, state)
            loss = F.cross_entropy(y_h, y.to(dev))
            optimizer.zero_grad()
            # scheduler.zero_grad()
            loss.backward()
            optimizer.step()
            # print('{} optim: {}'.format(epoch, optimizer.param_groups[0]['lr']))
            scheduler.step()
            # print('{} scheduler: {}'.format(epoch, scheduler.get_lr()[0]))
            total_train_loss += loss.item()
            total_train_correct += (y_h.argmax(-1) == y.cuda()).float().mean()
            count += 1
        average_train_loss = total_train_loss / count
        average_train_accuracy = total_train_correct / count
        print('{} optim: {}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
        # print('{} optim: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        # print('{} scheduler: {}'.format(epoch, scheduler.get_lr()[0]))
        losses.append(average_train_loss)

        print(f'epoch {epoch + 1} accuracies: \t train: {average_train_accuracy}\t loss: {average_train_loss}\t')
        # print(f'epoch {epoch + 1} accuracies: \t train: {average_train_accuracy}\t valid: {average_valid_accuracy}\t valid loss: {average_valid_loss}\t precision: {average_precision}\t recall: {average_recall}\t')
        # dataset.shuffle()

    # test mode
    # dataset.set_partition(dataset.test)
    model.eval()
    total_test_correct = 0
    # total_test_tp = 0
    # total_test_fp = 0
    # total_test_fn = 0
    count = 0
    for x, y, meta, meta_wi in dataset.test_dataset:
        state = None
        y_h, state = model(x.to(dev), meta, meta_wi, state)
        total_test_correct += (y_h.argmax(-1) == y.cuda()).float().mean()
        # total_test_tp += float((y_h.argmax(-1) == y.cuda() & (y.cuda() == 1)).float())
        # total_test_fp += float((y_h.argmax(-1) != y.cuda() & (y.cuda() == 0)).float())
        # total_test_fn += float((y_h.argmax(-1) != y.cuda() & (y.cuda() == 1)).float())
        count += 1
    average_test_accuracy = total_test_correct / count
    # average_precision = total_test_tp / (total_test_tp + total_test_fp)
    # average_recall = total_test_tp / (total_test_tp + total_test_fn)

    # print(f'test accuracy {average_test_accuracy} precision {average_precision} recall {average_recall}')
    print(f'test accuracy {average_test_accuracy}')

    return losses, (average_train_accuracy, average_test_accuracy)

def test_model(dataset, model, dev):
    model.eval()
    model.enable_explain()
    total_test_correct = 0
    count = 0
    output = []
    for x, y, meta, meta_wi in dataset.test_dataset:
        state = None
        y_h, state = model(x.to(dev), meta, meta_wi, state)
        total_test_correct += (y_h.argmax(-1) == y.cuda()).float().mean()
        output.extend(y_h.tolist())
        count += 1
    average_test_accuracy = total_test_correct / count
    print(f'test accuracy {average_test_accuracy}')
    return average_test_accuracy, output
