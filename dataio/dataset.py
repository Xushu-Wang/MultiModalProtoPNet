from dataio.genetics import GeneticDataset
from dataio.bio1mscan import Bio1MScan
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataset(cfg, log):
    if cfg.DATASET.NAME == 'multimodal':
        
        normalize = transforms.Normalize(
            mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN, 
            std=cfg.DATASET.IMAGE.TRANSFORM_STD
        )

        
        train_dataset = Bio1MScan(
            datapath = cfg.DATASET.GENETIC.TRAIN_PATH,
            imgpath=cfg.DATASET.IMAGE.TRAIN_DIR,
            img_transformation=transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE.SIZE, cfg.DATASET.IMAGE.SIZE)),
                transforms.ToTensor(),
                normalize]),
            genetic_transformation=cfg.DATASET.GENETIC.TRANSFORM,
            level=cfg.DATASET.GENETIC.TAXONOMY_NAME
        )
        
        
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=False)
        
        
        train_push_dataset = Bio1MScan(
            datapath = cfg.DATASET.GENETIC.TRAIN_PUSH_DIR,
            imgpath=cfg.DATASET.IMAGE.TRAIN_PUSH_DIR,
            img_transformation=transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE.SIZE, cfg.DATASET.IMAGE.SIZE)),
                transforms.ToTensor(),
                normalize]),
            genetic_transformation=cfg.DATASET.GENETIC.TRANSFORM,
            level=cfg.DATASET.GENETIC.TAXONOMY_NAME
        )
        
        
        train_push_loader = DataLoader(
            train_push_dataset, batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)
        
        test_dataset = Bio1MScan(
            datapath = cfg.DATASET.GENETIC.VALIDATION_PATH,
            imgpath=cfg.DATASET.IMAGE.TEST_DIR,
            img_transformation=transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE.SIZE, cfg.DATASET.IMAGE.SIZE)),
                transforms.ToTensor(),
                normalize,]),
            genetic_transformation=cfg.DATASET.GENETIC.TRANSFORM,
            level=cfg.DATASET.GENETIC.TAXONOMY_NAME,
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)
        
        
        return train_loader, train_push_loader, test_loader, 
    
    elif cfg.DATASET.NAME == 'genetics':
        train_dataset = GeneticDataset(cfg.DATASET.GENETIC.TRAIN_PATH,
                              cfg.DATASET.GENETIC.TRANSFORM, 
                              cfg.DATASET.GENETIC.TAXONOMY_NAME,
                              restraint=cfg.DATASET.GENETIC.RESTRAINT,
                              max_class_count=cfg.DATASET.NUM_CLASSES
                        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=False)
        
        train_push_dataset = GeneticDataset(cfg.DATASET.GENETIC.TRAIN_PUSH_DIR,
                                cfg.DATASET.GENETIC.TRANSFORM, 
                                cfg.DATASET.GENETIC.TAXONOMY_NAME,
                                restraint=cfg.DATASET.GENETIC.RESTRAINT,
                                max_class_count=cfg.DATASET.NUM_CLASSES
                            )
        
        train_push_loader = DataLoader(
            train_push_dataset, batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)

        validation_dataset = GeneticDataset(cfg.DATASET.GENETIC.VALIDATION_PATH, 
                              cfg.DATASET.GENETIC.TRANSFORM,
                              cfg.DATASET.GENETIC.TAXONOMY_NAME,
                              train_dataset.get_classes(cfg.DATASET.GENETIC.TAXONOMY_NAME)[0])
        
        
        validation_loader = DataLoader(
            validation_dataset, batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)
        
        return train_loader, train_push_loader, validation_loader
    
    
    elif cfg.DATASET.NAME == "cub" or cfg.DATASET.NAME == "bioscan":
        
        normalize = transforms.Normalize(
            mean=cfg.DATASET.IMAGE.TRANSFORM_MEAN, 
            std=cfg.DATASET.IMAGE.TRANSFORM_STD
        )

        # train set
        train_dataset = datasets.ImageFolder(
            cfg.DATASET.IMAGE.TRAIN_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE.SIZE, cfg.DATASET.IMAGE.SIZE)),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.DATASET.TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=False)

        # push set
        train_push_dataset = datasets.ImageFolder(
            cfg.DATASET.IMAGE.TRAIN_PUSH_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE.SIZE, cfg.DATASET.IMAGE.SIZE)),
                transforms.ToTensor(),
            ]))
        train_push_loader = DataLoader(
            train_push_dataset, batch_size=cfg.DATASET.TRAIN_PUSH_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)

        # test set
        test_dataset = datasets.ImageFolder(
            cfg.DATASET.IMAGE.TEST_DIR,
            transforms.Compose([
                transforms.Resize(size=(cfg.DATASET.IMAGE.SIZE, cfg.DATASET.IMAGE.SIZE)),
                transforms.ToTensor(),
                normalize,
            ]))

        test_loader = DataLoader(
            test_dataset, batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=False)
        
        return train_loader, train_push_loader, test_loader
    
    else:
        raise NotImplementedError
        
        
    

    
    
    
    

