from dataio.genetics import GeneticDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader



def get_dataset(cfg):
    if cfg.DATASET.NAME == 'bio1mscan':
        pass
    
    elif cfg.DATASET.NAME == 'genetics':
        return 
    
    elif cfg.DATASET.NAME == "cub":
        
        normalize = transforms.Normalize(
            mean=cfg.DATASET.TRANSFORM_MEAN, 
            std=cfg.DATASET.TRANSFORM_STD
        )

        # train set
        train_dataset = datasets.ImageFolder(
            cfg.DATASET.TRAIN_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=False)

        # push set
        train_push_dataset = datasets.ImageFolder(
            cfg.DATASET.TRAIN_PUSH_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                transforms.ToTensor(),
            ]))
        train_push_loader = DataLoader(
            train_push_dataset, batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)

        # test set
        test_dataset = datasets.ImageFolder(
            cfg.DATASET.TEST_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE_SIZE, cfg.DATASET.IMAGE_SIZE)),
                transforms.ToTensor(),
                normalize,
            ]))

        test_loader = DataLoader(
            test_dataset, batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)
        
        
        return train_loader, train_push_loader, test_loader
    
    else:
        raise NotImplementedError
        
        
    

    
    
    
    

