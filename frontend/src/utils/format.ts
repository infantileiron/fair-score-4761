import type {FrontierInterval} from '../types/domain';
export const percent=(v:number|null|undefined)=>v==null?'无分母':(v*100).toFixed(1)+'%';
export const date=(v:string)=>new Date(v).toLocaleString('zh-CN',{hour12:false});
export const short=(v:string)=>v.length>12?v.slice(0,12)+'…':v;
export const thresholdInterval=(i:FrontierInterval)=>`${i.lower_inclusive?'≥':'＞'}${i.lower} 且 ${i.upper_inclusive?'≤':'＜'}${i.upper}`;
