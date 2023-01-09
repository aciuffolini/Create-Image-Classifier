def save_check(model, in_args, train_data):

    print("saving model")
    if (in_args.save_dir is None):
        save_dir = 'checkpoint.pth'
    else:
        save_dir = in_args.save_dir

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': model,
                  'model': model,
                  'classifier': model.classifier,
                  'features': model.features,
                  'class_to_idx': train_data.class_to_idx,
                  'state_dict': model.state_dict()}


    torch.save(checkpoint, save_dir)
    return
