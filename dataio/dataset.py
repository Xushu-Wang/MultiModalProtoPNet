from dataio.genetics import GeneticDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader



def get_dataset(cfg, log):
    if cfg.DATASET.NAME == 'multimodal':
        pass
    
    elif cfg.DATASET.NAME == 'genetics':
            
        train_dataset = GeneticDataset(cfg.DATASET.TRAIN_PATH,
                              cfg.DATASET.TRANSFORM, 
                              cfg.DATASET.BIOSCAN.TAXONOMY_NAME)
        
        
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=False)
        
        
        validation_dataset = GeneticDataset(cfg.DATASET.TRAIN_PATH, 
                              cfg.DATASET.TRANSFORM,
                              cfg.DATASET.BIOSCAN.TAXONOMY_NAME,
                              train_dataset.get_classes(cfg.DATASET.BIOSCAN.TAXONOMY_NAME)[0])
        
        
        validation_loader = DataLoader(
            validation_dataset, batch_size=cfg.DATASET.VALIDATION_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)
        
        
        return train_loader, None, validation_loader
    
    
    elif cfg.DATASET.NAME == "cub" or cfg.DATASET.NAME == "bioscan":
        
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
        
        
    

    
    
    
    

