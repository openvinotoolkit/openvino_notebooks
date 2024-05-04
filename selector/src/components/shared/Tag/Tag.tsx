import './Tag.scss';

const sparkClassNames = {
  tag: 'spark-tag spark-focus-visible spark-focus-visible-self spark-focus-visible-snap',
  tagSizePrefix: 'spark-tag-size-',
  tagVariantPrefix: 'spark-tag-',
  tagRoundingPrefix: 'spark-tag-rounding-',
  tagThemePrefix: 'spark-tag-theme-',
};

type TagSize = 'small' | 'large';

type TagVariant = 'action' | 'primary' | 'secondary' | 'ghost';

type TagRounding = 'none' | 'semi-round' | 'fully-round';

type TagTheme = 'none' | 'classic' | 'geode' | 'moss' | 'rust' | 'coral' | 'cobalt' | 'daisy-tint1';

type TagProps = {
  text: string;
  size?: TagSize;
  variant?: TagVariant;
  rounding?: TagRounding;
  theme?: TagTheme;
};

export const Tag = (props: TagProps): JSX.Element => {
  const { text, size = 'small', variant = 'action', rounding = 'none', theme = 'none' } = props;
  const sizeClassName = `${sparkClassNames.tagSizePrefix}${size}`;
  const variantClassName = `${sparkClassNames.tagVariantPrefix}${variant}`;
  const roundingClassName = `${sparkClassNames.tagRoundingPrefix}${rounding}`;
  const themeClassName = `${sparkClassNames.tagThemePrefix}${theme}`;
  const classNames = [sparkClassNames.tag, sizeClassName, variantClassName, roundingClassName, themeClassName].join(
    ' '
  );
  return <span className={classNames}>{text}</span>;
};
